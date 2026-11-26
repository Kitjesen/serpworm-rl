"""Device-resident corotational beam/rigid/tendon/wheel physics core.

Beam matrices and spline frame follow BeamAdapter (SOFA, LGPL-2.1-or-later),
whose source evidence and license are archived in archive/reference/BeamAdapter.
No SOFA calls, NumPy mechanics, or device-to-host reads inside physics().
Supports arbitrary elements_per_strip (SOFA-free parametric topology),
frozen-tangent substep reuse (tangent_interval >= 1), and CUDA Graph capture
across both PyTorch and fused CUDA backends.
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from serpworm.paths import INPUTS_DIR
from serpworm.physics.topology import build_topology
from serpworm.physics.wheel_stiction import (
    arpc_normal_and_rut,
    arpc_tangential_contact,
    contact as implicit_wheel_contact,
)
from serpworm.physics.pipe_geometry import _tensor_any, eval_pipe_contact_torch, make_off_pipe

_CONTACT_MODEL_IDS = {
    'kelvin': 0,
    'arpc_hard': 1,
    'arpc_carpet': 2,
    'arpc_soft_soil': 3,
}

def cross(a,b): return torch.cross(a,b,dim=-1)
def norm(x): return torch.linalg.vector_norm(x,dim=-1,keepdim=True)
def unit(x): return x/norm(x).clamp_min(1e-15)
def mul(a,b):
    return torch.cat((a[...,3:]*b[...,:3]+b[...,3:]*a[...,:3]+cross(a[...,:3],b[...,:3]),a[...,3:]*b[...,3:]-(a[...,:3]*b[...,:3]).sum(-1,keepdim=True)),-1)
def conj(q): return torch.cat((-q[...,:3],q[...,3:]),-1)
def rotate(q,x):
    t=2*cross(q[...,:3],x)
    return x+q[...,3:]*t+cross(q[...,:3],t)
def logq(q):
    q=torch.where(q[...,3:]<0,-q,q)
    n=norm(q[...,:3]);return q[...,:3]*(2*torch.atan2(n,q[...,3:])/n.clamp_min(1e-15))
def stable_logq(q):
    """Opt-in quaternion log with the correct 2I derivative at identity."""
    q=torch.where(q[...,3:]<0,-q,q);v,w=q[...,:3],q[...,3:]
    n2=(v*v).sum(-1,keepdim=True);n=torch.sqrt(n2.clamp_min(1e-12));ws=w.clamp_min(1e-8)
    ratio=torch.where(n2<1e-12,2/ws-2*n2/(3*ws**3)+2*n2*n2/(5*ws**5),2*torch.atan2(n,w)/n)
    return v*ratio
def skew(v):
    x,y,z=v.unbind(-1);o=torch.zeros_like(x)
    return torch.stack((o,-z,y,z,o,-x,-y,x,o),-1).reshape(*v.shape[:-1],3,3)
def matrix(q):
    s=skew(q[...,:3]);eye=torch.eye(3,device=q.device,dtype=q.dtype)
    return eye+2*q[...,3:,None]*s+2*s@s
def align(q,t):
    ex=torch.zeros_like(t);ex[...,0]=1
    a=rotate(q,ex);c=(a*t).sum(-1,keepdim=True)
    r=unit(torch.cat((cross(a,t),1+c),-1))
    return mul(r,q)
def midpoint(p,q,L):
    ex=torch.zeros_like(p);ex[...,0]=1
    tangent=rotate(q,ex)
    p0,p3=p.unbind(-2);t0,t1=tangent.unbind(-2)
    p1=p0+L[...,None]*t0/3;p2=p3-L[...,None]*t1/3
    center=(p0+3*p1+3*p2+p3)/8
    t=unit(-.75*p0-.75*p1+.75*p2+.75*p3)
    r0=align(q[...,0,:],t);r1=align(q[...,1,:],t)
    r1=torch.where((r0*r1).sum(-1,keepdim=True)<0,-r1,r1)
    r=align(unit(r0+r1),t)
    compressed=(((p1-p0)*(p2-p1)).sum(-1)<0)&(norm(p3-p0)[...,0]<.4*L)
    return torch.where(compressed[...,None],(p0+p3)/2,center),torch.where(compressed[...,None],q[...,0,:],r)

def _unit_d(v,dv):
    n=norm(v);u=v/n.clamp_min(1e-15);eye=torch.eye(v.shape[-1],device=v.device,dtype=v.dtype)
    outer=torch.where((n>1e-15)[...,None],u[..., :,None]*u[...,None,:],0.)
    return u,(eye-outer)@dv/n.clamp_min(1e-15)[...,None]
def _qmatrix(q,left=True):
    v,w=q[...,:3],q[...,3:];eye=torch.eye(3,device=q.device,dtype=q.dtype)
    return torch.cat((torch.cat((w[...,None]*eye+(1 if left else -1)*skew(v),v[...,None]),-1),torch.cat((-v[...,None,:],w[...,None]),-1)),-2)
def _mul_d(a,da,b,db):return mul(a,b),_qmatrix(b,False)@da+_qmatrix(a)@db
def _omega_map(q):
    eye=torch.eye(3,device=q.device,dtype=q.dtype)
    return 2*torch.cat((q[...,3:,None]*eye+skew(q[...,:3]),-q[...,:3,None]),-1)
def _align_d(q,dq,t,dt):
    ex=torch.zeros_like(t);ex[...,0]=1;a=rotate(q,ex);da=-skew(a)@(_omega_map(q)@dq)
    raw=torch.cat((cross(a,t),1+(a*t).sum(-1,keepdim=True)),-1)
    draw=torch.cat((-skew(t)@da+skew(a)@dt,t[...,None,:]@da+a[...,None,:]@dt),-2)
    r,dr=_unit_d(raw,draw);return _mul_d(r,dr,q,dq)
def beam_strain_jacobian(p,q,offset,L):
    """Analytic du/d(world endpoint twist), including the spline-frame motion."""
    eye=torch.eye(3,device=p.device,dtype=p.dtype)
    W=torch.zeros((*p.shape,12),device=p.device,dtype=p.dtype);P=torch.zeros_like(W)
    W[...,0,:,3:6]=eye;W[...,1,:,9:]=eye
    P[...,0,:,:3]=eye;P[...,0,:,3:6]=-skew(offset[...,0,:]);P[...,1,:,6:9]=eye;P[...,1,:,9:]=-skew(offset[...,1,:])
    dq=.5*torch.cat((q[...,3:,None]*eye-skew(q[...,:3]),-q[...,:3][...,None,:]),-2)@W
    ex=torch.zeros_like(p);ex[...,0]=1;ti=rotate(q,ex);dti=-skew(ti)@W
    p0,p3=p.unbind(-2);t0,t1=ti.unbind(-2);p1=p0+L[...,None]*t0/3;p2=p3-L[...,None]*t1/3
    tangent_raw=-.75*p0-.75*p1+.75*p2+.75*p3
    dtr=1.5*(P[...,1,:,:]-P[...,0,:,:])-.25*L[...,None,None]*(dti[...,0,:,:]+dti[...,1,:,:])
    t,dt=_unit_d(tangent_raw,dtr)
    r0,dr0=_align_d(q[...,0,:],dq[...,0,:,:],t,dt);r1,dr1=_align_d(q[...,1,:],dq[...,1,:,:],t,dt)
    sign=torch.where((r0*r1).sum(-1,keepdim=True)<0,-1.,1.)
    middle,dmiddle=_unit_d(r0+sign*r1,dr0+sign[...,None]*dr1);r,dr=_align_d(middle,dmiddle,t,dt)
    compressed=(((p1-p0)*(p2-p1)).sum(-1)<0)&(norm(p3-p0)[...,0]<.4*L)
    r=torch.where(compressed[...,None],q[...,0,:],r);dr=torch.where(compressed[...,None,None],dq[...,0,:,:],dr)
    B=_omega_map(r)@dr
    center=(p0+3*p1+3*p2+p3)/8;center=torch.where(compressed[...,None],(p0+p3)/2,center)
    C=.5*(P[...,0,:,:]+P[...,1,:,:])+L[...,None,None]/8*(-skew(t0)@W[...,0,:,:]+skew(t1)@W[...,1,:,:])
    C=torch.where(compressed[...,None,None],.5*(P[...,0,:,:]+P[...,1,:,:]),C)
    Rt=matrix(r).transpose(-1,-2);phi=stable_logq(mul(conj(r)[...,None,:].expand_as(q),q))
    Jp=Rt[...,None,:,:]@(P-C[...,None,:,:]+skew(p-center[...,None,:])@B[...,None,:,:])
    theta2=(phi*phi).sum(-1,keepdim=True);theta=torch.sqrt(theta2.clamp_min(1e-12))
    coeff=torch.where(theta2<1e-8,1/12+theta2/720+theta2*theta2/30240,(1-.5*theta/torch.tan(.5*theta))/theta2.clamp_min(1e-12))
    S=skew(phi);A=eye-.5*S+coeff[...,None]*(S@S)
    Jr=A@Rt[...,None,:,:]@(W-B[...,None,:,:])
    return torch.cat((Jp,Jr),-2).reshape(*p.shape[:-2],12,12)

def local_matrices_torch(lengths,p,device='cpu',dtype=torch.float64,thin_shell_compat=False):
    """Device-native vectorized local stiffness K and consistent mass M.

    When ``thin_shell_compat=True``, applies wide thin-ribbon Kirchhoff-Love plate
    flexural stiffening ``E / (1 - nu^2)`` to out-of-plane bending ``Iy`` and Vlasov
    restrained-warping torsional correction for high width-to-thickness steel strips.
    """
    L=torch.as_tensor(lengths,device=device,dtype=dtype).reshape(-1)
    n=L.numel()
    w=float(p['strip_width_m']);t=float(p['strip_thickness_m']);A=w*t
    nu=float(p['poisson_ratio'])
    Iy=w*t**3/12;Iz=t*w**3/12;J=w*t**3*(1/3-.21*t/w*(1-t**4/(12*w**4)))
    E=float(p['youngs_modulus_pa']);G=E/(2*(1+nu));rho=float(p['steel_density_kg_m3'])
    E_y=E/(1.0-nu*nu) if thin_shell_compat else E
    J_eff=J*(1.0+(w*w)/(12.0*0.145*0.145*(1.0-nu))) if thin_shell_compat else J
    K=torch.zeros((n,12,12),device=device,dtype=dtype)
    M=torch.zeros((n,12,12),device=device,dtype=dtype)
    idx=torch.arange(6,device=device)
    kd=torch.stack((E*A/L,12*E*Iz/L**3,12*E_y*Iy/L**3,G*J_eff/L,4*E_y*Iy/L,4*E*Iz/L),-1)
    K[:,idx,idx]=kd;K[:,idx+6,idx+6]=kd
    a42=-6*E_y*Iy/L**2;a51=6*E*Iz/L**2
    K[:,4,2]=a42;K[:,2,4]=a42;K[:,5,1]=a51;K[:,1,5]=a51
    K[:,10,8]=-a42;K[:,8,10]=-a42;K[:,11,7]=-a51;K[:,7,11]=-a51
    kc=torch.stack((-kd[:,0],-kd[:,1],-kd[:,2],-kd[:,3],2*E_y*Iy/L,2*E*Iz/L),-1)
    K[:,idx+6,idx]=kc;K[:,idx,idx+6]=kc
    K[:,7,5]=-a51;K[:,5,7]=-a51;K[:,8,4]=-a42;K[:,4,8]=-a42
    K[:,10,2]=a42;K[:,2,10]=a42;K[:,11,1]=a51;K[:,1,11]=a51

    o=torch.ones_like(L)
    md=torch.stack((o/3,13/35+6*Iz/(5*A*L**2),13/35+6*Iy/(5*A*L**2),o*(Iy+Iz)/(3*A),L**2/105+2*Iy/(15*A),L**2/105+2*Iz/(15*A)),-1)
    M[:,idx,idx]=md;M[:,idx+6,idx+6]=md
    m42=-11*L/210-Iy/(10*A*L);m51=11*L/210+Iz/(10*A*L)
    M[:,4,2]=m42;M[:,2,4]=m42;M[:,5,1]=m51;M[:,1,5]=m51
    M[:,10,8]=-m42;M[:,8,10]=-m42;M[:,11,7]=-m51;M[:,7,11]=-m51
    mc=torch.stack((o/6,9/70-6*Iz/(5*A*L**2),9/70-6*Iy/(5*A*L**2),o*(Iy+Iz)/(6*A),-L**2/140-Iy/(30*A),-L**2/140-Iz/(30*A)),-1)
    M[:,idx+6,idx]=mc;M[:,idx,idx+6]=mc
    c15=13*L/420-Iz/(10*A*L);c24=-13*L/420+Iy/(10*A*L)
    M[:,7,5]=c15;M[:,5,7]=c15;M[:,8,4]=c24;M[:,4,8]=c24
    M[:,10,2]=-c24;M[:,2,10]=-c24;M[:,11,1]=-c15;M[:,1,11]=-c15
    M*=(rho*A*L)[:,None,None]
    return K,M

def local_matrices(lengths,p):
    """NumPy wrapper around local_matrices_torch for offline compatibility."""
    K,M=local_matrices_torch(lengths,p,device='cpu',dtype=torch.float64)
    return K.numpy(),M.numpy()

def pack_beam_matrix(mat:torch.Tensor)->torch.Tensor:
    """Pack a (..., 12, 12) beam matrix into its 16 non-redundant scalars for fused CUDA V2."""
    return torch.stack((
        mat[...,0,0],mat[...,1,1],mat[...,2,2],
        mat[...,3,3],mat[...,4,4],mat[...,5,5],
        mat[...,1,5],mat[...,2,4],
        mat[...,0,6],mat[...,1,7],mat[...,2,8],
        mat[...,3,9],mat[...,4,10],mat[...,5,11],
        mat[...,1,11],mat[...,2,10],
    ),dim=-1).contiguous()

class GPUCore:
    def __init__(self,count=1,batch=1,dt=.00025,device='cuda',dtype=torch.float64,sofa_mass_compat=False,legacy_rolling=False,yaw_motor_speed=None,ground_contact=True,budget=False,strain_jacobian=False,plate_solver='legacy',inputs_dir=None,hardware=None,elements_per_strip=None,tangent_interval=1):
        if count<1 or batch<1 or not 0<dt<=.002:raise ValueError('Invalid topology/time step')
        if plate_solver not in ('legacy','segment_lu','fused_cuda'):raise ValueError('Invalid plate solver')
        if int(tangent_interval)<1:raise ValueError('tangent_interval must be >= 1')
        self.count,self.batch,self.dt,self.device,self.dtype=count,batch,dt,torch.device(device),dtype
        self.plate_solver=plate_solver
        self.tangent_interval=int(tangent_interval)
        self.sofa_mass_compat=sofa_mass_compat
        self.legacy_rolling=legacy_rolling or sofa_mass_compat
        self.ground_contact=bool(ground_contact)
        self.budget_enabled=bool(budget)
        self.strain_jacobian=bool(strain_jacobian)
        if self.tangent_interval>1 and (plate_solver!='segment_lu' or self.strain_jacobian or self.budget_enabled):
            raise ValueError('tangent_interval > 1 requires plate_solver="segment_lu" without strain_jacobian/budget')
        self.last_budget=None
        self._capturing_graph=False
        self._capture_has_gpipe=False
        self._capture_has_legacy_wall=True
        self.hardware=hardware
        if hardware is not None:
            wmu=float(hardware.wheel_mu)
            ykp=float(hardware.yaw_kp_nm_rad);ykd=float(hardware.yaw_kd_nms_rad)
            ytlim=float(hardware.yaw_torque_limit_nm)
            yslim=float(hardware.yaw_speed_limit_rad_s) if yaw_motor_speed is None else float(yaw_motor_speed)
            stlim=float(hardware.servo_torque_limit_nm);sslim=float(hardware.servo_speed_limit_rad_s)
            self.stop_gap=float(hardware.plate_stop_gap_m)
            self.tendon_crank_mode=(getattr(hardware,'tendon_model','legacy_wrap')=='crank_calibrated')
            self.horn_radius=float(getattr(hardware,'tendon_horn_radius_m',0.0235))
            self.guide_dist=float(getattr(hardware,'tendon_guide_dist_m',0.0350))
            self.tendon_slack=float(getattr(hardware,'tendon_slack_m',0.0))
            self.tendon_stiffness=float(getattr(hardware,'tendon_stiffness_n_m',2000.0))
            self.tendon_damping=float(getattr(hardware,'tendon_damping_ns_m',0.5))
            self.tendon_y_m=float(getattr(hardware,'tendon_y_m',0.0285))
            self.contact_model=str(getattr(hardware,'contact_model','kelvin'))
            self.implicit_contact=bool(getattr(hardware,'implicit_contact',False))
            self.ground_kn=float(getattr(hardware,'ground_kn_n_m',5000.0))
            self.ground_cn=float(getattr(hardware,'ground_cn_ns_m',8.0))
            self.ground_hc_k=float(getattr(hardware,'ground_hc_k',1.6e5))
            self.ground_hc_alpha=float(getattr(hardware,'ground_hc_alpha',3.5))
            self.ground_soil_exp=float(getattr(hardware,'ground_soil_exp',1.5))
            self.ground_soil_unload_ratio=float(getattr(hardware,'ground_soil_unload_ratio',1.0))
            self.ground_rut_relax_m=float(getattr(hardware,'ground_rut_relax_m',0.025))
            self.ground_rut_lat_k=float(getattr(hardware,'ground_rut_lat_k',400.0))
            self.wheel_mu_lat_ratio=float(getattr(hardware,'wheel_mu_lat_ratio',1.0))
            self.wheel_rolling=float(getattr(hardware,'wheel_rolling',0.06))
            self.wheel_mass_kg=float(getattr(hardware,'wheel_mass_kg',0.009))
            self.wheel_patch_radius=float(getattr(hardware,'wheel_patch_radius_m',0.0))
            self.wheel_pivot_mu=float(getattr(hardware,'wheel_pivot_mu',0.65))
            self.ratchet_backlash=float(getattr(hardware,'ratchet_backlash_rad',0.0))
            self.wheel_static_mu_ratio=float(getattr(hardware,'wheel_static_mu_ratio',1.0))
            self.rayleigh_alpha=float(getattr(hardware,'rayleigh_alpha',0.05))
            self.rayleigh_beta=float(getattr(hardware,'rayleigh_beta',0.002))
            self.tendon_capstan_mu_theta=float(getattr(hardware,'tendon_capstan_mu_theta',0.0))
            self.yaw_static_friction=float(getattr(hardware,'yaw_static_friction_nm',0.0))
            self.servo_deadband=float(getattr(hardware,'servo_deadband_rad',0.0))
            self._plate_masses_cfg=getattr(hardware,'plate_masses_kg',None)
            self.strip_thin_shell_compat=bool(getattr(hardware,'strip_thin_shell_compat',False))
            self.strip_prestress_frac=float(getattr(hardware,'strip_prestress_frac',0.0))
            self.strip_node_contact=bool(getattr(hardware,'strip_node_contact',False))
        else:
            wmu=.8;ykp=2.;ykd=.03;ytlim=.5
            yslim=-1. if yaw_motor_speed is None else float(yaw_motor_speed)
            stlim=.5;sslim=2.;self.stop_gap=.075
            self.tendon_crank_mode=False
            self.horn_radius=.0235;self.guide_dist=.035;self.tendon_slack=0.
            self.tendon_stiffness=2000.;self.tendon_damping=.5;self.tendon_y_m=.0285
            self.contact_model='kelvin';self.implicit_contact=False
            self.ground_kn=5000.;self.ground_cn=8.
            self.ground_hc_k=1.6e5;self.ground_hc_alpha=3.5;self.ground_soil_exp=1.5
            self.ground_soil_unload_ratio=1.;self.ground_rut_relax_m=.025;self.ground_rut_lat_k=400.
            self.wheel_mu_lat_ratio=1.;self.wheel_rolling=.06;self.wheel_mass_kg=.009
            self.wheel_patch_radius=0.;self.wheel_pivot_mu=.65;self.ratchet_backlash=0.
            self.wheel_static_mu_ratio=1.0
            self.rayleigh_alpha=0.05;self.rayleigh_beta=0.002
            self.tendon_capstan_mu_theta=0.0;self.yaw_static_friction=0.0;self.servo_deadband=0.0
            self._plate_masses_cfg=None
            self.strip_thin_shell_compat=False
            self.strip_prestress_frac=0.0
            self.strip_node_contact=False
        if self.contact_model not in _CONTACT_MODEL_IDS:
            raise ValueError(f'Unknown contact_model: {self.contact_model!r}')
        self.contact_model_id=_CONTACT_MODEL_IDS[self.contact_model]
        if self.tangent_interval>1 and self.implicit_contact:
            raise ValueError('tangent_interval > 1 is incompatible with implicit_contact=True')
        self.yaw_motor_speed=None if yslim<=0 else yslim
        self.wheel_mu=torch.full((batch,1,1),wmu,device=self.device,dtype=dtype)
        self.mass_scale=torch.ones((batch,),device=self.device,dtype=dtype)
        self.yaw_kp=torch.full((batch,),ykp,device=self.device,dtype=dtype)
        self.yaw_kd=torch.full((batch,),ykd,device=self.device,dtype=dtype)
        self.yaw_torque_limit=torch.full((batch,),ytlim,device=self.device,dtype=dtype)
        self.yaw_speed_limit=torch.full((batch,),yslim,device=self.device,dtype=dtype)
        self.servo_torque_limit=torch.full((batch,),stlim,device=self.device,dtype=dtype)
        self.servo_speed_limit=torch.full((batch,),sslim,device=self.device,dtype=dtype)
        # Per-env ARPC terrain properties: [wheel_rolling, hc_k, soil_unload_ratio, rut_lat_k, mu_lat_ratio]
        self.env_terrain=torch.empty((batch,5),device=self.device,dtype=dtype)
        self.env_terrain[:,0]=self.wheel_rolling
        self.env_terrain[:,1]=self.ground_hc_k
        self.env_terrain[:,2]=self.ground_soil_unload_ratio
        self.env_terrain[:,3]=self.ground_rut_lat_k
        self.env_terrain[:,4]=self.wheel_mu_lat_ratio
        # Per-env contact/tendon/gearbox dynamics: [backlash, static_mu_r, patch_r, capstan_mt, slack, yaw_sf, sv_db]
        self.env_dynamics=torch.empty((batch,7),device=self.device,dtype=dtype)
        self.env_dynamics[:,0]=self.ratchet_backlash
        self.env_dynamics[:,1]=self.wheel_static_mu_ratio
        self.env_dynamics[:,2]=self.wheel_patch_radius
        self.env_dynamics[:,3]=self.tendon_capstan_mu_theta
        self.env_dynamics[:,4]=self.tendon_slack
        self.env_dynamics[:,5]=self.yaw_static_friction
        self.env_dynamics[:,6]=self.servo_deadband
        # Per-env per-plate asymmetric rigid mass (B, 2*count) in kg
        if self._plate_masses_cfg is not None and len(self._plate_masses_cfg)>=2*count:
            pm_base=torch.tensor(self._plate_masses_cfg[:2*count],device=self.device,dtype=dtype)
        else:
            pm_base=torch.full((2*count,),0.28,device=self.device,dtype=dtype)
        self.plate_masses_base=pm_base
        self.plate_masses=pm_base[None,:].expand(batch,-1).clone()
        # Per-env structural Rayleigh stiffness damping beta (B,) and ground surface geometry (B, 5):
        # ground_geom = [slope_x, slope_y, wave_amp_m, wave_kx, wave_ky] (all zeros => exact flat z=0)
        self.rayleigh_beta_env=torch.full((batch,),self.rayleigh_beta,device=self.device,dtype=dtype)
        self.ground_geom=torch.zeros((batch,5),device=self.device,dtype=dtype)
        # Per-env bilateral pipe/corridor wall geometry: [y_center, y_allow, x_entry, x_exit, k_wall]
        # Default k_wall=0 & y_allow=1e6 disables lateral walls completely (zero overhead for open terrain).
        self.wall_cfg=torch.zeros((batch,5),device=self.device,dtype=dtype)
        self.wall_cfg[:,1]=1e6
        self.wall_cfg[:,2]=1e6
        self.wall_cfg[:,3]=-1e6
        self.last_wall_force=torch.zeros((batch,2*count),device=self.device,dtype=dtype)
        # Generalized analytical pipe/bend/S-curve/sharp-90-L-duct config (B, 12) + 40-strip node telemetry
        self.pipe_cfg=torch.tensor(make_off_pipe(),device=self.device,dtype=dtype)[None,:].repeat(batch,1)
        data_dir=Path(inputs_dir) if inputs_dir is not None else INPUTS_DIR
        self.parameters=json.loads((data_dir/'parameters.json').read_text(encoding='utf-8'))
        custom_topo=False
        if hardware is not None:
            if getattr(hardware,'strip_width_m',None) is not None:
                self.parameters['strip_width_m']=float(hardware.strip_width_m)
                custom_topo=True
            if getattr(hardware,'strip_thickness_m',None) is not None:
                self.parameters['strip_thickness_m']=float(hardware.strip_thickness_m)
                custom_topo=True
            if getattr(hardware,'stress_free_bow_m',None) is not None:
                self.parameters['stress_free_bow_m']=float(hardware.stress_free_bow_m)
                custom_topo=True
        npz_path=data_dir/f'topology_{count}.npz'
        if (not custom_topo) and (elements_per_strip is None or int(elements_per_strip)==16) and npz_path.is_file():
            raw=dict(np.load(npz_path))
        else:
            E_req=16 if elements_per_strip is None else int(elements_per_strip)
            raw=build_topology(count=count,elements_per_strip=E_req,parameters=self.parameters)
        self.nodes=self.tensor(raw['strip_nodes'],torch.long);self.edges=self.tensor(raw['edges'],torch.long)
        self.S,self.N=self.nodes.shape;self.E=self.N-1;self.P=2*count;self.D=len(raw['rest'])
        if plate_solver in ('segment_lu','fused_cuda'):
            expected=np.repeat(np.arange(self.P).reshape(count,2),self.S//count,axis=0)
            if self.S%count or not np.array_equal(raw['strip_nodes'][:,[0,-1]],expected):
                raise ValueError('segment_lu/fused_cuda requires strips grouped by adjacent segment plate pairs')
            self.segment_ids=torch.arange(count,device=self.device)
        self.lengths=self.tensor(raw['lengths']).reshape(self.S,self.E)
        self.transforms=self.tensor(raw['transforms']);self.mounts=self.tensor(raw['wheel_mounts'])[torch.arange(self.P,device=self.device)%2]
        self.anchors=self.tensor(raw['anchors']);self.guides=self.tensor(raw['guides'])
        if abs(self.tendon_y_m-0.0285)>1e-9:
            yt=self.tendon_y_m
            self.anchors=self.anchors.clone();self.guides=self.guides.clone()
            y_vec=self.tensor([yt,yt,-yt,-yt])
            self.anchors[:,1]=y_vec;self.guides[:,1]=y_vec
        self.x=self.tensor(raw['rest'])[None].repeat(batch,1,1);self.rest=self.x.clone();self.v=torch.zeros((batch,self.D,6),device=self.device,dtype=dtype)
        self.servo=torch.zeros((batch,2*count),device=self.device,dtype=dtype);self.target=self.servo.clone();self.yaw=torch.zeros((batch,max(0,count-1)),device=self.device,dtype=dtype)
        self.spin=torch.zeros((batch,self.P,2),device=self.device,dtype=dtype);self.angle=self.spin.clone();self.memory=torch.zeros((batch,self.P,2,3),device=self.device,dtype=dtype)
        self.ratchet_play=torch.zeros((batch,self.P,2),device=self.device,dtype=dtype)
        self.soil_sinkage=torch.zeros((batch,self.P,2),device=self.device,dtype=dtype)
        self._plate_K_contact=torch.zeros((batch,self.P,21),device=self.device,dtype=dtype)
        self._plate_K_contact_mat=torch.zeros((batch,self.P,6,6),device=self.device,dtype=dtype)
        self._tri6_r,self._tri6_c=torch.tril_indices(6,6,device=self.device)
        self.tension=torch.zeros((batch,count,4),device=self.device,dtype=dtype);self.support=self.spin.clone()
        self.last_slip=torch.zeros((batch,self.P,2,2),device=self.device,dtype=dtype)
        self.last_yaw_torque=torch.zeros((batch,max(0,count-1)),device=self.device,dtype=dtype)
        self.last_yaw_torque_cap=torch.full((batch,max(0,count-1)),ytlim,device=self.device,dtype=dtype)
        self.last_strip_wall_force=torch.zeros((batch,self.S),device=self.device,dtype=dtype)
        self.last_strip_min_z=torch.full((batch,self.S),0.10,device=self.device,dtype=dtype)
        k,m=local_matrices_torch(raw['lengths'],self.parameters,device=self.device,dtype=dtype,thin_shell_compat=self.strip_thin_shell_compat)
        self.K=k.reshape(self.S,self.E,12,12);self.M=m.reshape(self.S,self.E,12,12)
        self.rest_local=self.frames()[0]
        if self.strip_prestress_frac>0.0:
            pf=float(max(0.0,min(1.0,self.strip_prestress_frac)))
            flat_local=torch.zeros_like(self.rest_local)
            flat_local[...,0]=-0.5*self.lengths
            flat_local[...,6]=0.5*self.lengths
            self.rest_local=(1.0-pf)*self.rest_local+pf*flat_local
        self.endpoint_ids=self.nodes[:,[0,-1]].reshape(-1)
        self.interior_ids=self.nodes[:,1:-1].reshape(-1)
        self.plate_dofs=(self.nodes[:,[0,-1],None]*6+torch.arange(6,device=self.device)).reshape(self.S,12)
        self.scatter_mat=(self.plate_dofs[:,:,None]*(self.P*6)+self.plate_dofs[:,None,:]).reshape(-1)
        self.gravity=torch.zeros_like(self.v);self.gravity[...,2]=-9.81
        self.connector_arm=self.tensor([-.058,0.,-.0010452742534])
        self.plate_inertia_diag=self.tensor([.0016,.0009,.0009])*.28
        self.plate_inertia=torch.diag(self.plate_inertia_diag)
        self.failed=torch.zeros((),device=self.device,dtype=torch.bool)
        self.graph=None
        self.graph_steps=0
        self._tangent_cache=None
        self._frozen_r=None
        self._frozen_offset=None
        self._substep_counter=0
        self._fused_ext=None
        if plate_solver=='fused_cuda':
            if self.device.type!='cuda' or self.dtype!=torch.float64:
                raise ValueError('fused_cuda requires CUDA float64')
            if self.strain_jacobian or self.legacy_rolling or self.sofa_mass_compat or self.budget_enabled:
                raise ValueError('fused_cuda supports standard implicit-stiction non-budget physics')
            from serpworm.physics.fused_cuda import get_extension
            self._fused_ext=get_extension()
            self._nodes_i32=self.nodes.to(torch.int32).contiguous()
            A_loc=(1+dt*self.rayleigh_alpha)*self.M+(dt*self.rayleigh_beta+dt*dt)*self.K
            self._K_pack=pack_beam_matrix(self.K)
            self._M_pack=pack_beam_matrix(self.M)
            self._A_pack=pack_beam_matrix(A_loc)
            self._rest_local_c=self.rest_local.contiguous()
            self._transforms_c=self.transforms.contiguous()
            self._lengths_c=self.lengths.contiguous()
            self._mounts_c=self.mounts.contiguous()
            self._anchors_c=self.anchors.contiguous()
            self._guides_c=self.guides.contiguous()
            self._plate_ext=torch.zeros((batch,self.P,6),device=self.device,dtype=dtype)
            self._strip_buf=torch.empty((max(1,self.E-2),78,batch*self.S),device=self.device,dtype=dtype)

    def tensor(self,x,dtype=None):return torch.as_tensor(x,device=self.device,dtype=dtype or self.dtype)

    def ground_height_and_normal(self,xy):
        """Evaluate per-env ground height z_g(x,y) and local unit normal up_g(x,y) for xy shape (B, ..., 2)."""
        shape_b=(self.batch,)+(1,)*(xy.ndim-2)
        sx=self.ground_geom[:,0].reshape(shape_b)
        sy=self.ground_geom[:,1].reshape(shape_b)
        amp=self.ground_geom[:,2].reshape(shape_b)
        kx=self.ground_geom[:,3].reshape(shape_b)
        ky=self.ground_geom[:,4].reshape(shape_b)
        wx=xy[...,0];wy=xy[...,1]
        s_x=torch.sin(kx*wx);c_x=torch.cos(kx*wx)
        s_y=torch.sin(ky*wy);c_y=torch.cos(ky*wy)
        z_g=sx*wx+sy*wy+amp*s_x*c_y
        dz_dx=sx+amp*kx*c_x*c_y
        dz_dy=sy-amp*ky*s_x*s_y
        up_raw=torch.stack((-dz_dx,-dz_dy,torch.ones_like(z_g)),dim=-1)
        return z_g,unit(up_raw)

    def align_pose_to_ground(self,mask):
        """Stress-free rigid rotation & lift placing reset envs flush onto their sloped/undulating ground_geom."""
        sx=self.ground_geom[:,0];sy=self.ground_geom[:,1];amp=self.ground_geom[:,2]
        rx=torch.atan(sy);ry=-torch.atan(sx)
        cx,sx2=torch.cos(0.5*rx),torch.sin(0.5*rx)
        cy,sy2=torch.cos(0.5*ry),torch.sin(0.5*ry)
        qs=torch.stack((sx2*cy,cx*sy2,-sx2*sy2,cx*cy),dim=-1)
        com=self.x[:,:self.P,:3].mean(dim=1,keepdim=True)
        rel=self.x[...,:3]-com
        qs_exp=qs[:,None,:].expand(-1,self.D,-1)
        pos_rot=rotate(qs_exp,rel)+com
        z_com,_=self.ground_height_and_normal(com[...,:2])
        pos_rot[...,2]=pos_rot[...,2]+z_com+amp[:,None].abs()
        quat_rot=unit(mul(qs_exp,self.x[...,3:]))
        x_new=torch.cat((pos_rot,quat_rot),dim=-1)
        self.x.copy_(torch.where(mask[:,None,None],x_new,self.x))

    def frames(self,with_jacobian=False,return_frame=False):
        xx=self.x[:,self.edges].reshape(self.batch,self.S,self.E,2,7)
        q=mul(xx[...,3:],self.transforms[...,3:].expand(self.batch,-1,-1,-1,-1))
        offset=rotate(xx[...,3:],self.transforms[...,:3].expand(self.batch,-1,-1,-1,-1));p=xx[...,:3]+offset
        center,r=midpoint(p,q,self.lengths)
        inv=conj(r)[...,None,:].expand(-1,-1,-1,2,-1)
        logarithm=stable_logq if self.strain_jacobian else logq
        u=torch.cat((rotate(inv,p-center[...,None,:]),logarithm(mul(inv,q))),-1).reshape(self.batch,self.S,self.E,12)
        rt=matrix(r).transpose(-1,-2)[...,None,:,:].expand(-1,-1,-1,2,-1,-1)
        zero=torch.zeros_like(rt)
        trans=torch.cat((torch.cat((rt,-rt@skew(offset)),-1),torch.cat((zero,rt),-1)),-2)
        T=torch.zeros((self.batch,self.S,self.E,12,12),device=self.device,dtype=self.dtype)
        T[...,:6,:6]=trans[...,0,:,:];T[...,6:,6:]=trans[...,1,:,:]
        if with_jacobian:return u,T,beam_strain_jacobian(p,q,offset,self.lengths)
        if return_frame:return u,T,r,offset
        return u,T

    def beam(self,return_frame=False):
        if self.strain_jacobian:
            u,T,J=self.frames(with_jacobian=True)
            elastic=-(J.transpose(-1,-2)@(self.K@(u-self.rest_local)[...,None]))[...,0]
            K=J.transpose(-1,-2)@self.K@J;M=T.transpose(-1,-2)@self.M@T
            return self.scatter_vector(elastic),K,M
        if return_frame:
            u,T,r,offset=self.frames(return_frame=True)
        else:
            u,T=self.frames()
        elastic=-(T.transpose(-1,-2)@(self.K@(u-self.rest_local)[...,None]))[...,0]
        K=T.transpose(-1,-2)@self.K@T;M=T.transpose(-1,-2)@self.M@T
        f=torch.zeros_like(self.v)
        f.index_add_(1,self.edges.reshape(-1),elastic.reshape(self.batch,-1,6))
        if return_frame:return f,K,M,r,offset
        return f,K,M

    def beam_rhs_only(self,h,r_f=None,offset_f=None):
        """Fast local RHS assembly (M @ tvg - h * K @ du) co-aligned with cached corotational frame."""
        if r_f is None or offset_f is None:
            u,T=self.frames()
            vg=(self.v+h*self.gravity)[:,self.edges].reshape(self.batch,self.S,self.E,12,1)
            floc=self.M@(T@vg)-h*(self.K@(u-self.rest_local)[...,None])
            world_f=(T.transpose(-1,-2)@floc)[...,0]
        else:
            xx=self.x[:,self.edges].reshape(self.batch,self.S,self.E,2,7)
            q=mul(xx[...,3:],self.transforms[...,3:].expand(self.batch,-1,-1,-1,-1))
            offset=rotate(xx[...,3:],self.transforms[...,:3].expand(self.batch,-1,-1,-1,-1));p=xx[...,:3]+offset
            ex=torch.zeros_like(p);ex[...,0]=1;tangent=rotate(q,ex)
            p0,p3=p.unbind(-2);t0,t1=tangent.unbind(-2)
            p1=p0+self.lengths[...,None]*t0/3;p2=p3-self.lengths[...,None]*t1/3
            center=(p0+3*p1+3*p2+p3)/8
            compressed=(((p1-p0)*(p2-p1)).sum(-1)<0)&(norm(p3-p0)[...,0]<.4*self.lengths)
            center=torch.where(compressed[...,None],(p0+p3)/2,center)
            r_exp=r_f[...,None,:].expand(-1,-1,-1,2,-1);inv_f=conj(r_exp)
            u=torch.cat((rotate(inv_f,p-center[...,None,:]),logq(mul(inv_f,q))),-1).reshape(self.batch,self.S,self.E,12)
            vg2=(self.v+h*self.gravity)[:,self.edges].reshape(self.batch,self.S,self.E,2,6)
            v_rel=vg2[...,:3]+cross(vg2[...,3:],offset_f)
            tvg=torch.cat((rotate(inv_f,v_rel),rotate(inv_f,vg2[...,3:])),-1).reshape(self.batch,self.S,self.E,12,1)
            floc=(self.M@tvg-h*(self.K@(u-self.rest_local)[...,None])).reshape(self.batch,self.S,self.E,2,6)
            fw=rotate(r_exp,floc[...,:3])
            tw=rotate(r_exp,floc[...,3:])+cross(offset_f,fw)
            world_f=torch.cat((fw,tw),-1)
        out=torch.zeros_like(self.v)
        out.index_add_(1,self.edges.reshape(-1),world_f.reshape(self.batch,-1,6))
        return out

    def external(self):
        h=self.dt;x=self.x[:,:self.P];v=self.v[:,:self.P];q=x[...,3:];R=matrix(q);f=torch.zeros_like(self.v)
        radius=self.horn_radius;guide=self.guide_dist
        chord=torch.sqrt(guide**2+radius**2-2*guide*radius*torch.cos(self.servo)).clamp_min(1e-9)
        if self.tendon_crank_mode:
            lever=(radius*guide*torch.sin(self.servo).abs()/chord).clamp_max(radius)
        else:
            alpha=math.acos(radius/guide)
            lever=torch.where(self.servo<=alpha,radius*guide*torch.sin(self.servo)/chord,radius)
        cap_mt=self.env_dynamics[:,3:4].clamp(0.0,0.6)
        sv_db=self.env_dynamics[:,6:7].clamp_min(0.0)
        load=self.tension.reshape(self.batch,2*self.count,2).sum(-1)*lever*torch.exp(cap_mt)
        s_max=self.servo_speed_limit[:,None]*h
        raw_err=self.target-self.servo
        eff_err=torch.sign(raw_err)*(raw_err.abs()-sv_db).clamp_min(0.0)
        inc=eff_err.clamp(-s_max,s_max);inc=inc*torch.where(inc>0,(1-load/self.servo_torque_limit[:,None]).clamp(0,1),1.)
        sv_vel=inc/h
        self.servo.add_(inc)
        chord=torch.sqrt(guide**2+radius**2-2*guide*radius*torch.cos(self.servo))
        if self.tendon_crank_mode:
            takeup=chord-(guide-radius)
        else:
            alpha=math.acos(radius/guide)
            takeup=torch.where(self.servo<=alpha,chord,math.sqrt(guide**2-radius**2)+radius*(self.servo-alpha))-(guide-radius)
        ar=rotate(q[:,0::2,None,:].expand(-1,-1,4,-1),self.anchors.expand(self.batch,self.count,-1,-1))
        br=rotate(q[:,1::2,None,:].expand(-1,-1,4,-1),self.guides.expand(self.batch,self.count,-1,-1))
        delta=x[:,1::2,None,:3]+br-x[:,0::2,None,:3]-ar;length=norm(delta);direction=delta/length.clamp_min(1e-15)
        va=v[:,0::2,None,:3]+cross(v[:,0::2,None,3:].expand_as(ar),ar);vb=v[:,1::2,None,:3]+cross(v[:,1::2,None,3:].expand_as(br),br)
        env_slack=self.env_dynamics[:,4:5,None]
        rest=(.101+env_slack)-takeup.reshape(self.batch,self.count,2).repeat_interleave(2,-1)
        extension=length[...,0]-rest;speed=((vb-va)*direction).sum(-1)
        cap_mt_t=cap_mt[:,:,None]
        c_eff=self.tendon_damping+12.0*cap_mt_t
        ten_raw=torch.where(extension>=0,(self.tendon_stiffness*extension+c_eff*speed).clamp_min(0),0.)
        sv_vel_t=sv_vel.reshape(self.batch,self.count,2).repeat_interleave(2,-1)
        pull_frac=torch.tanh(sv_vel_t.clamp_min(0.0)/0.35)
        self.tension.copy_(ten_raw*torch.exp(-cap_mt_t*pull_frac))
        force=self.tension[...,None]*direction
        f[:,:self.P:2,:3]+=force.sum(-2);f[:,1:self.P:2,:3]-=force.sum(-2)
        f[:,:self.P:2,3:]+=cross(ar,force).sum(-2);f[:,1:self.P:2,3:]-=cross(br,force).sum(-2)
        sep=x[:,1::2,:3]-x[:,0::2,:3];axis=-R[:,0::2,:,0];gap=(sep*axis).sum(-1)
        push=50000*(self.stop_gap-gap).clamp_min(0)[...,None]*axis
        f[:,1:self.P:2,:3]+=push;f[:,:self.P:2,:3]-=push;f[:,:self.P:2,3:]-=cross(sep,push)
        if self.count>1:
            a=slice(1,self.P-1,2);b=slice(2,self.P,2)
            arm=rotate(q[:,a],self.connector_arm.expand(self.batch,self.count-1,-1))
            sep=x[:,b,:3]-x[:,a,:3];err=sep-arm
            relv=v[:,b,:3]-v[:,a,:3]-cross(v[:,a,3:],sep);force=-10000*err-10*relv
            f[:,b,:3]+=force;f[:,a,:3]-=force;f[:,a,3:]-=cross(sep,force)
            relative=R[:,a].transpose(-1,-2)@R[:,b];yaw=torch.atan2(relative[...,1,0],relative[...,0,0]);axis=R[:,a,:,2]
            omega=v[:,b,3:]-v[:,a,3:];rate=(omega*axis).sum(-1,keepdim=True)
            kp=self.yaw_kp[:,None,None];kd=self.yaw_kd[:,None,None];tlim=self.yaw_torque_limit[:,None,None];slim=self.yaw_speed_limit[:,None,None]
            yaw_sf=self.env_dynamics[:,5:6,None].clamp_min(0.0)
            raw_yt=kp*(self.yaw-yaw)[...,None]-kd*rate
            yaw_torque=(torch.sign(raw_yt)*(raw_yt.abs()-yaw_sf).clamp_min(0.0)).clamp(-tlim,tlim)
            speed_en=slim>0
            cap=torch.where(speed_en,tlim*torch.where(yaw_torque*rate>0,(1-rate.abs()/slim.clamp_min(1e-9)).clamp(0,1),1.),tlim)
            yaw_torque=torch.maximum(torch.minimum(yaw_torque,cap),-cap)
            self.last_yaw_torque_cap=cap[...,0]
            self.last_yaw_torque=yaw_torque[...,0]
            torque=yaw_torque*axis+20*cross(R[:,b,:,2],axis)-.1*(omega-axis*rate)
            f[:,b,3:]+=torque;f[:,a,3:]-=torque
        self.internal_force=f.sum(1)[:,:3]
        self.internal_torque=(cross(self.x[...,:3],f[...,:3])+f[...,3:]).sum(1)
        if self.budget_enabled:self.last_internal_load=f.clone()
        axle=R[...,1];up0=torch.zeros_like(axle);up0[...,2]=1
        rim=-.018*unit(up0-axle*axle[...,2:])
        offset=rotate(q[...,None,:].expand(-1,-1,2,-1),self.mounts.expand(self.batch,-1,-1,-1))
        arm=offset+rim[...,None,:]
        velocity=v[...,None,:3]+cross(v[...,None,3:].expand_as(offset),arm)
        contact_pos=x[...,None,:3]+arm
        z_g,up_g=self.ground_height_and_normal(contact_pos[...,:2])
        penetration=(z_g-contact_pos[...,2])*up_g[...,2]
        axle_w=axle[:,:,None,:].expand_as(up_g)
        forward=unit(cross(up_g,axle_w));lateral=cross(up_g,forward)
        vn=(velocity*up_g).sum(-1)
        basis=torch.stack((forward,lateral),-2)
        tangent_v=torch.einsum('bpskc,bpsc->bpsk',basis,velocity);original_mem=torch.einsum('bpskc,bpsc->bpsk',basis,self.memory)
        inertia=.5*self.wheel_mass_kg*.018**2
        w_roll=self.env_terrain[:,0:1,None]
        hc_k=self.env_terrain[:,1:2,None]
        unload_r=self.env_terrain[:,2:3,None]
        rut_lat_k=self.env_terrain[:,3:4,None]
        mu_lat_r=self.env_terrain[:,4:5,None]
        if self.contact_model_id==0 and not self.implicit_contact:
            active=(penetration>0)&self.ground_contact
            normal=torch.where(active,(self.ground_kn*penetration-self.ground_cn*vn).clamp_min(0),0.)
            if self.legacy_rolling:
                mem=original_mem;gain=2000*h+8
                resistance=torch.minimum(w_roll*normal*.018+2e-6*self.spin,inertia*self.spin/h)
                base=-2000*mem-gain*tangent_v
                omega=((self.spin-h*(.018*base[...,0]+resistance)/inertia)/(1+h*.018**2*gain/inertia)).clamp_min(0)
                tang=base.clone();tang[...,0]+=gain*.018*omega
                mag=norm(tang);cap=self.wheel_mu*normal;sliding=mag[...,0]>cap
                tang=tang*torch.minimum(torch.ones_like(mag),cap[...,None]/mag.clamp_min(1e-15))
                free=self.spin-h*(.018*tang[...,0]+resistance)/inertia;omega=free.clamp_min(0)
                reaction=inertia*(omega-free)/h-resistance
                slip=tangent_v.clone();slip[...,0]-=.018*omega
                mem=torch.where(sliding[...,None],-(tang+8*slip)/2000,mem+h*slip)
            else:
                tang,omega,mem,reaction,slip=implicit_wheel_contact(tangent_v,self.spin,original_mem,normal,h,mu=self.wheel_mu,rolling=w_roll,wheel_mass=self.wheel_mass_kg)
            self.last_slip=slip
            self.memory.copy_(torch.where(active[...,None],torch.einsum('bpsk,bpskc->bpsc',mem,basis),0.))
            self.spin.copy_(torch.where(active,omega,self.spin*max(0,1-h*2e-6/inertia)));self.angle.add_(h*self.spin);self.support.copy_(normal)
            force=torch.einsum('bpsk,bpskc->bpsc',tang,basis)+normal[...,None]*up_g;force=torch.where(active[...,None],force,0.)
            rt=cross(rim[...,None,:].expand_as(force),force)
            torque=cross(offset,force)+rt-axle[...,None,:]*(rt*axle[...,None,:]).sum(-1,keepdim=True)+torch.where(active,reaction,0.)[...,None]*axle[...,None,:]
            f[:,:self.P,:3]+=force.sum(-2);f[:,:self.P,3:]+=torque.sum(-2)
        else:
            if self.contact_model_id==0:
                raw_n=self.ground_kn*penetration-self.ground_cn*vn
                active=(penetration>0.0)&(raw_n>0.0)&self.ground_contact
                zeros=torch.zeros_like(penetration)
                normal=torch.where(active,raw_n,zeros)
                kn_tan=torch.where(active,torch.full_like(penetration,self.ground_kn),zeros)
                cn_tan=torch.where(active,torch.full_like(penetration,self.ground_cn),zeros)
                new_rut=zeros
            else:
                if self.ground_contact:
                    v_horiz=torch.hypot(tangent_v[...,0],tangent_v[...,1])
                    normal,new_rut,kn_tan,cn_tan=arpc_normal_and_rut(
                        penetration,vn,v_horiz,self.soil_sinkage,h,
                        hc_k,self.ground_hc_alpha,self.ground_soil_exp,
                        unload_r,self.ground_rut_relax_m,
                    )
                else:
                    zeros=torch.zeros_like(penetration)
                    normal=new_rut=kn_tan=cn_tan=zeros
                active=normal>0.0
            self.soil_sinkage.copy_(new_rut)
            wz_plate=v[...,None,5].expand_as(normal)
            mu_fwd=self.wheel_mu
            mu_lat=self.wheel_mu*mu_lat_r
            env_backlash=self.env_dynamics[:,0:1,None]
            env_static_mu_r=self.env_dynamics[:,1:2,None]
            env_patch_r=self.env_dynamics[:,2:3,None]
            tang,omega,mem,new_play,reaction,slip,tau_z,ct_fwd_tan,ct_lat_tan,c_piv_tan=arpc_tangential_contact(
                tangent_v,self.spin,original_mem,self.ratchet_play,normal,new_rut,wz_plate,h,
                radius=0.018,mu_fwd=mu_fwd,mu_lat=mu_lat,rolling=w_roll,
                backlash_rad=env_backlash,patch_radius=env_patch_r,
                pivot_mu=self.wheel_pivot_mu,rut_lat_k=rut_lat_k,
                static_mu_ratio=env_static_mu_r,
                wheel_mass=self.wheel_mass_kg,
            )
            self.last_slip=slip
            self.memory.copy_(torch.where(active[...,None],torch.einsum('bpsk,bpskc->bpsc',mem,basis),0.0))
            self.spin.copy_(torch.where(active,omega,self.spin*max(0.0,1.0-h*2e-6/inertia)))
            self.angle.add_(h*self.spin)
            self.ratchet_play.copy_(torch.where(active,new_play,self.ratchet_play))
            self.support.copy_(normal)
            force=torch.einsum('bpsk,bpskc->bpsc',tang,basis)+normal[...,None]*up_g;force=torch.where(active[...,None],force,0.0)
            rt=cross(rim[...,None,:].expand_as(force),force)
            torque=cross(offset,force)+rt-axle[...,None,:]*(rt*axle[...,None,:]).sum(-1,keepdim=True)+torch.where(active,reaction,0.0)[...,None]*axle[...,None,:]
            torque[...,2]+=torch.where(active,tau_z,0.0)
            f[:,:self.P,:3]+=force.sum(-2);f[:,:self.P,3:]+=torque.sum(-2)
            if self.implicit_contact:
                zeros=torch.zeros_like(normal)
                c0=torch.where(active,ct_fwd_tan,zeros)
                c1=torch.where(active,ct_lat_tan,zeros)
                c2=torch.where(active,cn_tan,zeros)
                cp=torch.where(active,c_piv_tan,zeros)
                k2=torch.where(active,kn_tan,zeros)
                u0=forward
                u1=lateral
                u2=up_g
                j0=torch.cat((u0,cross(arm,u0)),dim=-1)
                j1=torch.cat((u1,cross(arm,u1)),dim=-1)
                j2=torch.cat((u2,cross(arm,u2)),dim=-1)
                f_comp=(
                    (c0*tangent_v[...,0])[...,None]*j0
                    +(c1*tangent_v[...,1])[...,None]*j1
                    +(c2*vn)[...,None]*j2
                ).sum(-2)
                f_comp[...,5]+=(cp*wz_plate).sum(-1)
                f[:,:self.P]+=f_comp
                w0=h*c0;w1=h*c1;w2=h*c2+(h*h)*k2;wp=h*cp
                K_c6=(
                    w0[...,None,None]*(j0[...,:,None]*j0[...,None,:])
                    +w1[...,None,None]*(j1[...,:,None]*j1[...,None,:])
                    +w2[...,None,None]*(j2[...,:,None]*j2[...,None,:])
                ).sum(-3)
                K_c6[...,5,5]+=wp.sum(-1)
                self._plate_K_contact_mat.copy_(K_c6)
                self._plate_K_contact.copy_(K_c6[...,self._tri6_r,self._tri6_c])
            else:
                self._plate_K_contact_mat.zero_()
                self._plate_K_contact.zero_()
        # Per-env generalized analytical pipe (straight / 90-deg arc / S-curve / sharp 90-deg L-duct)
        # or legacy wall_cfg[B, 5] fallback (exact parity with fused_physics.cu)
        use_gpipe=self.pipe_cfg[:,0]>0.5
        has_gpipe=self._capture_has_gpipe if self._capturing_graph else _tensor_any(use_gpipe)
        if has_gpipe:
            f_gp,k_gp,fn_gp=eval_pipe_contact_torch(
                x[:,:,:3],v[:,:,:3],self.pipe_cfg,
                node_radius=0.074,dt=h,implicit_contact=self.implicit_contact,k_scale=1.0,c_wall=18.0,
            )
            f[:,:self.P,:3]+=torch.where(use_gpipe[:,None,None],f_gp,0.0)
            if self.implicit_contact:
                self._plate_K_contact_mat[:,:,:3,:3]+=torch.where(use_gpipe[:,None,None,None],k_gp,0.0)
                self._plate_K_contact.copy_(self._plate_K_contact_mat[...,self._tri6_r,self._tri6_c])
        else:
            fn_gp=torch.zeros((self.batch,self.P),device=self.device,dtype=self.dtype)

        has_legacy=self._capture_has_legacy_wall if self._capturing_graph else _tensor_any(~use_gpipe)
        if has_legacy:
            y_c=self.wall_cfg[:,0:1];y_al=self.wall_cfg[:,1:2]
            x_en=self.wall_cfg[:,2:3];x_ex=self.wall_cfg[:,3:4];k_w=self.wall_cfg[:,4:5]
            px_w=x[:,:,0];py_w=x[:,:,1];vx_w=v[:,:,0];vy_w=v[:,:,1]
            in_fun=(px_w<=x_en+0.08)&(px_w>x_en)
            in_cor=(px_w<=x_en)&(px_w>=x_ex)
            act_w=(~use_gpipe[:,None])&(k_w>0.0)&(y_al<10.0)&(in_fun|in_cor)
            u_fun=((x_en+0.08-px_w)/0.08).clamp(0.0,1.0)
            y_lim=torch.where(in_fun,y_al+0.055*(1.0-u_fun),y_al)
            dy_w=py_w-y_c
            pen_w=(dy_w.abs()-y_lim).clamp_min(0.0)
            sgn_w=torch.where(dy_w>=0.0,1.0,-1.0)
            hit_w=act_w&(pen_w>0.0)
            c_wall=18.0;mu_wall=0.35;vx_reg=0.015
            fn_w=torch.where(hit_w,(k_w*pen_w+c_wall*(vy_w*sgn_w)).clamp_min(0.0),0.0)
            pos_fn=fn_w>0.0
            th_vx=torch.tanh(vx_w/vx_reg)
            f[:,:self.P,1]+=-sgn_w*fn_w
            f[:,:self.P,0]+=-mu_wall*fn_w*th_vx
            if self.implicit_contact:
                f[:,:self.P,1]+=torch.where(pos_fn,c_wall*vy_w,0.0)
                ct_x=torch.where(pos_fn,(mu_wall*fn_w/vx_reg)*(1.0-th_vx*th_vx).clamp_min(0.0),0.0)
                f[:,:self.P,0]+=ct_x*vx_w
                self._plate_K_contact_mat[:,:,1,1]+=torch.where(pos_fn,h*c_wall+(h*h)*k_w,0.0)
                self._plate_K_contact_mat[:,:,0,0]+=h*ct_x
                self._plate_K_contact.copy_(self._plate_K_contact_mat[...,self._tri6_r,self._tri6_c])
            self.last_wall_force.copy_(torch.where(use_gpipe[:,None],fn_gp,fn_w))
        else:
            self.last_wall_force.copy_(fn_gp)
        if self.budget_enabled:
            self.last_ground_load=f-self.last_internal_load
            self.last_ground_force=force.clone()
            self.last_contact_position=x[...,None,:3]+arm
        return f

    def mass_action(self,velocity,M=None,pm=None):
        if M is None:_,_,M=self.beam()
        if pm is None:pm=self.plate_mass()
        ve=velocity[:,self.edges].reshape(self.batch,self.S,self.E,12)
        p=self.scatter_vector((M@ve[...,None])[...,0])
        p[:,:self.P]+=(pm@velocity[:,:self.P,...,None])[...,0]
        return p

    def project_resultant(self,load,position=None):
        if position is None:position=self.x[...,:3]
        return torch.cat((load[...,:3].sum(1),(cross(position,load[...,:3])+load[...,3:]).sum(1)),-1)

    def budget_state(self):
        u,T=self.frames();M=T.transpose(-1,-2)@self.M@T
        pm=self.plate_mass();p=self.mass_action(self.v,M,pm)
        du=u-self.rest_local
        kinetic=.5*(self.v*p).sum((1,2))
        beam=.5*(du*(self.K@du[...,None])[...,0]).sum((1,2,3))
        R=matrix(self.x[:,:self.P,3:]);axle=R[...,1]
        wheel_inertia=.5*self.wheel_mass_kg*.018**2
        spin_momentum=-(wheel_inertia*self.spin[...,None]*axle[...,None,:]).sum((1,2))
        spin_energy=.5*wheel_inertia*(self.spin**2).sum((1,2))
        ar=rotate(self.x[:,:self.P:2,None,3:].expand(-1,-1,4,-1),self.anchors.expand(self.batch,self.count,-1,-1))
        br=rotate(self.x[:,1:self.P:2,None,3:].expand(-1,-1,4,-1),self.guides.expand(self.batch,self.count,-1,-1))
        delta=self.x[:,1:self.P:2,None,:3]+br-self.x[:,:self.P:2,None,:3]-ar
        radius,guide=self.horn_radius,self.guide_dist
        chord=torch.sqrt(guide**2+radius**2-2*guide*radius*torch.cos(self.servo))
        if self.tendon_crank_mode:
            takeup=chord-(guide-radius)
        else:
            alpha=math.acos(radius/guide)
            takeup=torch.where(self.servo<=alpha,chord,math.sqrt(guide**2-radius**2)+radius*(self.servo-alpha))-(guide-radius)
        extension=norm(delta)[...,0]-(.101+self.env_dynamics[:,4:5,None])+takeup.reshape(self.batch,self.count,2).repeat_interleave(2,-1)
        tendon=.5*self.tendon_stiffness*extension.clamp_min(0).square().sum((1,2))
        gap=((self.x[:,1:self.P:2,:3]-self.x[:,:self.P:2,:3])*(-R[:,:self.P:2,:,0])).sum(-1)
        stop=25000*(self.stop_gap-gap).clamp_min(0).square().sum(1)
        connector=torch.zeros_like(stop)
        if self.count>1:
            a=slice(1,self.P-1,2);b=slice(2,self.P,2)
            arm=rotate(self.x[:,a,3:],self.connector_arm.expand(self.batch,self.count-1,-1))
            err=self.x[:,b,:3]-self.x[:,a,:3]-arm
            connector=5000*err.square().sum((1,2))+20*(1-(R[:,a,:,2]*R[:,b,:,2]).sum(-1)).sum(1)
        offset=rotate(self.x[:,:self.P,None,3:].expand(-1,-1,2,-1),self.mounts.expand(self.batch,-1,-1,-1))
        up=torch.zeros_like(axle);up[...,2]=1
        rim=-.018*unit(up-axle*axle[...,2:])
        penetration=-(self.x[:,:self.P,None,2]+offset[...,2]+rim[...,None,2])
        if self.contact_model_id==0:
            normal_storage=0.5*self.ground_kn*penetration.clamp_min(0).square().sum((1,2))*self.ground_contact
        else:
            normal_storage=(self.ground_hc_k/(self.ground_soil_exp+1.0))*penetration.clamp_min(0).pow(self.ground_soil_exp+1.0).sum((1,2))*self.ground_contact
        tangential_storage=1000*self.memory.square().sum((1,2,3))*self.ground_contact
        gravity_force=self.mass_action(self.gravity,M,pm)
        gravity_potential=-(self.x[...,:3]*gravity_force[...,:3]).sum((1,2))
        resultant=self.project_resultant(p)
        energies=dict(kinetic=kinetic,beam=beam,tendon=tendon,stop=stop,
            connector=connector,gravity=gravity_potential,wheel_spin=spin_energy,
            contact_normal=normal_storage,contact_tangent=tangential_storage)
        return dict(momentum=resultant,spin_momentum=spin_momentum,
            energy=energies,total_energy=sum(energies.values()),takeup=takeup)

    def finish_budget(self,before,old_x,old_v,vnew,ext,elastic,K,M,pm):
        h=self.dt;after=self.budget_state()
        p0=self.mass_action(old_v,M,pm);p1=self.mass_action(vnew,M,pm)
        ve=vnew[:,self.edges].reshape(self.batch,self.S,self.E,12)
        kv=self.scatter_vector((K@ve[...,None])[...,0])
        gravity=self.mass_action(self.gravity,M,pm)
        rayleigh=-self.rayleigh_alpha*p1-self.rayleigh_beta_env[:,None,None]*kv
        implicit=-h*kv
        compat=torch.zeros_like(ext)
        if self.sofa_mass_compat:
            vv=(old_v-self.gravity)[:,self.edges].reshape(self.batch,self.S,self.E,12)
            compat=.05*self.scatter_vector((M@vv[...,None])[...,0])
        loads=dict(beam=elastic,internal=self.last_internal_load,
            ground=self.last_ground_load,gravity=gravity,rayleigh=rayleigh,
            implicit_stiffness=implicit,compatibility=compat)
        impulses={name:h*self.project_resultant(value,old_x) for name,value in loads.items()}
        expected=sum(impulses.values())
        frozen_delta=self.project_resultant(p1-p0,old_x)
        actual_delta=after['momentum']-before['momentum']
        spin_delta=after['spin_momentum']-before['spin_momentum']
        ground_world_torque=cross(self.last_contact_position,self.last_ground_force).sum((1,2))
        ground_total=h*torch.cat((self.last_ground_force.sum((1,2)),ground_world_torque),-1)
        with_spin=actual_delta.clone();with_spin[:,3:]+=spin_delta
        expected_with_spin=expected-impulses['ground']+ground_total
        internal_impulse=impulses['beam']+impulses['internal']
        loss_rayleigh=-h*(vnew*rayleigh).sum((1,2))
        dv=vnew-old_v
        loss_implicit_kinetic=.5*(dv*self.mass_action(dv,M,pm)).sum((1,2))
        stiffness_work=h*(vnew*implicit).sum((1,2))
        takeup_delta=after['takeup']-before['takeup']
        horn_work=(self.tension.reshape(self.batch,2*self.count,2).sum(-1)*takeup_delta).sum(1)
        work={name:h*(vnew*value).sum((1,2)) for name,value in loads.items()}
        kinetic_frozen=.5*((vnew*p1)-(old_v*p0)).sum((1,2))
        self.last_budget=dict(before=before,after=after,impulse=impulses,
            frozen_equation_residual=frozen_delta-expected,
            frame_transport=actual_delta-frozen_delta,
            momentum_residual=actual_delta-expected,
            momentum_with_wheel_spin_residual=with_spin-expected_with_spin,
            noninternal_balance_with_wheel_spin_residual=with_spin-(expected_with_spin-internal_impulse),
            internal_resultant=self.project_resultant(elastic+self.last_internal_load,old_x),
            generalized_work=work,rayleigh_loss=loss_rayleigh,
            implicit_kinetic_loss=loss_implicit_kinetic,
            implicit_stiffness_work=stiffness_work,horn_boundary_work_estimate=horn_work,
            frozen_kinetic_work_residual=kinetic_frozen+loss_implicit_kinetic-sum(work.values()),
            partial_energy_residual=after['total_energy']-before['total_energy']-horn_work+loss_rayleigh)

    def scatter_vector(self,fe):
        out=torch.zeros_like(self.v);out.index_add_(1,self.edges.reshape(-1),fe.reshape(self.batch,-1,6));return out

    def plate_mass(self):
        m_bp=self.plate_masses*self.mass_scale[:,None]
        m=torch.zeros((self.batch,self.P,6,6),device=self.device,dtype=self.dtype)
        m[...,:3,:3]=m_bp[...,None,None]*torch.eye(3,device=self.device,dtype=self.dtype)
        m[...,3:,3:]=(m_bp/0.28)[...,None,None]*self.plate_inertia
        return m

    def _strip_interior_contact(self,h:float):
        """Exact implicit ground & 8-type pipe-wall contact on all S*(E-1) steel-strip interior nodes."""
        is_pipe_env=self.pipe_cfg[:,0]>0.5
        has_pipe=self._capture_has_gpipe if self._capturing_graph else _tensor_any(is_pipe_env)
        if (not self.strip_node_contact) and (not has_pipe):
            return None,None
        ids=self.nodes[:,1:-1].reshape(-1)
        pos=self.x[:,ids,:3]
        vel=self.v[:,ids,:3]
        B,M,_=pos.shape
        f_tot=torch.zeros((B,M,3),device=self.device,dtype=self.dtype)
        k_tot=torch.zeros((B,M,3,3),device=self.device,dtype=self.dtype)
        # 1. Strip-to-ground contact protection (exact ground_geom surface)
        z_g,up_g=self.ground_height_and_normal(pos[...,:2])
        clr_g=(pos[...,2]-z_g)*up_g[...,2]
        self.last_strip_min_z.copy_(clr_g.reshape(B,self.S,self.E-1).amin(-1))
        if self.ground_contact:
            from serpworm.physics.pipe_geometry import _accumulate_surface_contact
            pen_g=(0.002-clr_g).clamp_min(0.0)
            if not self.strip_node_contact:
                pen_g=torch.where(is_pipe_env[:,None],pen_g,torch.zeros_like(pen_g))
            fn_g=torch.zeros((B,M),device=self.device,dtype=self.dtype)
            k_sg=torch.full((B,1),2500.0,device=self.device,dtype=self.dtype)
            _accumulate_surface_contact(vel,pen_g,up_g,k_sg,8.0,0.35,h,True,f_tot,k_tot,fn_g)
        # 2. Strip-to-pipe-wall contact (straight / 90-deg arc / S-curve / sharp 90-deg L-duct)
        if has_pipe:
            f_p,k_p,fn_p=eval_pipe_contact_torch(
                pos,vel,self.pipe_cfg,node_radius=0.002,dt=h,implicit_contact=True,k_scale=0.20,c_wall=4.0,
            )
            f_tot.add_(f_p)
            k_tot.add_(k_p)
            self.last_strip_wall_force.copy_(fn_p.reshape(B,self.S,self.E-1).sum(-1))
        else:
            self.last_strip_wall_force.zero_()
        return f_tot.reshape(B,self.S,self.E-1,3),k_tot.reshape(B,self.S,self.E-1,3,3)

    def solve(self,A,rhs,plate,cache_factors=False,strip_K=None):
        """Batched block Thomas elimination, then a 6*(2*count) plate Schur solve."""
        diag=A[:,:,:-1,6:,6:]+A[:,:,1:,:6,:6];lower=A[:,:,1:-1,6:,:6];upper=A[:,:,1:-1,:6,6:]
        if strip_K is not None:
            diag=diag.clone()
            diag[...,:3,:3]+=strip_K
        r=torch.zeros((self.batch,self.S,self.E-1,6,13),device=self.device,dtype=self.dtype)
        r[...,0]=rhs[:,self.nodes[:,1:-1]]
        r[:,:,0,:,1:7]=A[:,:,0,6:,:6];r[:,:,-1,:,7:]=A[:,:,-1,:6,6:]
        ds=[];rs=[];factors=[];status=[]
        for i in range(self.E-1):
            d=diag[:,:,i];b=r[:,:,i]
            if i:
                factor=lower[:,:,i-1]@ds[-1];d=d-factor@upper[:,:,i-1];b=b-factor@rs[-1]
                factors.append(factor)
            inv,info=torch.linalg.inv_ex(d,check_errors=False);ds.append(inv);rs.append(b);status.append((info!=0).any())
        solution=[None]*(self.E-1);solution[-1]=ds[-1]@rs[-1]
        for i in range(self.E-3,-1,-1):solution[i]=ds[i]@(rs[i]-upper[:,:,i]@solution[i+1])
        z=torch.stack(solution,2)
        endA=torch.zeros((self.batch,self.S,12,12),device=self.device,dtype=self.dtype)
        endA[...,:6,:6]=A[:,:,0,:6,:6];endA[...,6:,6:]=A[:,:,-1,6:,6:]
        left=A[:,:,0,:6,6:];right=A[:,:,-1,6:,:6]
        endA[...,:6,:]-=left@z[:,:,0,:,1:];endA[...,6:,:]-=right@z[:,:,-1,:,1:]
        endr=-torch.cat(((left@z[:,:,0,:,0,None])[...,0],(right@z[:,:,-1,:,0,None])[...,0]),-1)
        P=self.P*6;globalA=torch.zeros((self.batch,P*P),device=self.device,dtype=self.dtype)
        globalA.index_add_(1,self.scatter_mat,endA.reshape(self.batch,-1));globalA=globalA.reshape(self.batch,P,P)
        for i in range(self.P):globalA[:,6*i:6*i+6,6*i:6*i+6]+=plate[:,i]
        globalr=rhs[:,:self.P].reshape(self.batch,P).clone();globalr.index_add_(1,self.plate_dofs.reshape(-1),endr.reshape(self.batch,-1))
        if self.plate_solver=='segment_lu':
            blocks=globalA.reshape(self.batch,self.count,12,self.count,12).permute(0,1,3,2,4)
            local=blocks[:,self.segment_ids,self.segment_ids].reshape(self.batch*self.count,12,12)
            if cache_factors:
                lu,piv,info=torch.linalg.lu_factor_ex(local,check_errors=False)
                vp=torch.linalg.lu_solve(lu,piv,globalr.reshape(self.batch*self.count,12,1)).reshape(self.batch,P,1)
                self._tangent_cache=(ds,factors,upper,z[...,1:],left,right,lu,piv)
            else:
                vp,info=torch.linalg.solve_ex(local,globalr.reshape(self.batch*self.count,12,1),check_errors=False)
                vp=vp.reshape(self.batch,P,1)
        elif self.batch<=8:
            vp,info=torch.linalg.solve_ex(globalA,globalr[...,None],check_errors=False)
        else:
            chunks=[torch.linalg.solve_ex(globalA[i:i+8],globalr[i:i+8,:,None],check_errors=False) for i in range(0,self.batch,8)]
            vp=torch.cat([s.result for s in chunks],0);info=torch.cat([s.info for s in chunks],0)
        vp=vp[...,0]
        self.failed.logical_or_(torch.stack(status).any()|(info!=0).any())
        vi=z[...,0]-(z[...,1:]@vp[:,self.plate_dofs][...,None,:,None])[...,0]
        out=torch.empty_like(rhs);out[:,:self.P]=vp.reshape(self.batch,self.P,6);out[:,self.interior_ids]=vi.reshape(self.batch,-1,6)
        return out

    def _solve_cached(self,rhs):
        """Fast single-RHS vector forward/back substitution using cached block-LU & Schur factors."""
        ds,factors,upper,z_bound,left,right,lu,piv=self._tangent_cache
        r_int=rhs[:,self.nodes[:,1:-1],:,None]
        rs=[]
        for i in range(self.E-1):
            b=r_int[:,:,i]
            if i:b=b-factors[i-1]@rs[-1]
            rs.append(b)
        sol=[None]*(self.E-1);sol[-1]=ds[-1]@rs[-1]
        for i in range(self.E-3,-1,-1):sol[i]=ds[i]@(rs[i]-upper[:,:,i]@sol[i+1])
        z0=torch.stack(sol,2)[...,0]
        endr=-torch.cat(((left@z0[:,:,0,:,None])[...,0],(right@z0[:,:,-1,:,None])[...,0]),-1)
        P=self.P*6
        globalr=rhs[:,:self.P].reshape(self.batch,P).clone()
        globalr.index_add_(1,self.plate_dofs.reshape(-1),endr.reshape(self.batch,-1))
        vp=torch.linalg.lu_solve(lu,piv,globalr.reshape(self.batch*self.count,12,1)).reshape(self.batch,P)
        vi=z0-(z_bound@vp[:,self.plate_dofs][...,None,:,None])[...,0]
        out=torch.empty_like(rhs);out[:,:self.P]=vp.reshape(self.batch,self.P,6);out[:,self.interior_ids]=vi.reshape(self.batch,-1,6)
        return out

    def physics(self,reuse_tangent=False):
        if self.budget_enabled:
            before=self.budget_state();old_x=self.x[...,:3].clone();old_v=self.v.clone()
        h=self.dt;ext=self.external();pm=self.plate_mass()
        if reuse_tangent and self._tangent_cache is not None:
            rhs=self.beam_rhs_only(h,self._frozen_r,self._frozen_offset)+h*ext
            rhs[:,:self.P]+=(pm@(self.v[:,:self.P]+h*self.gravity[:,:self.P])[...,None])[...,0]
            vnew=self._solve_cached(rhs)
        else:
            if self.tangent_interval>1:
                elastic,K,M,self._frozen_r,self._frozen_offset=self.beam(return_frame=True)
            else:
                elastic,K,M=self.beam()
            vg=(self.v+h*self.gravity)[:,self.edges].reshape(self.batch,self.S,self.E,12)
            rhs=self.scatter_vector((M@vg[...,None])[...,0])+h*(ext+elastic)
            if self.sofa_mass_compat:
                diff=(self.v-self.gravity)[:,self.edges].reshape(self.batch,self.S,self.E,12)
                rhs+=h*.05*self.scatter_vector((M@diff[...,None])[...,0])
            rhs[:,:self.P]+=(pm@(self.v[:,:self.P]+h*self.gravity[:,:self.P])[...,None])[...,0]
            strip_f,strip_K=self._strip_interior_contact(h)
            if strip_f is not None:
                rhs[:,self.interior_ids,:3]+=h*strip_f.reshape(self.batch,-1,3)
            A=(1+h*self.rayleigh_alpha)*M+(h*self.rayleigh_beta_env[:,None,None,None,None]+h*h)*K
            plate_lhs=(1+h*self.rayleigh_alpha)*pm+(self._plate_K_contact_mat if self.implicit_contact else 0.)
            vnew=self.solve(A,rhs,plate_lhs,cache_factors=(self.tangent_interval>1),strip_K=strip_K)
        self.v.copy_(vnew);self.x[...,:3].add_(h*vnew[...,:3])
        omega=.5*h*vnew[...,3:];theta=norm(omega)
        exp=torch.cat((omega*torch.sinc(theta/math.pi),torch.cos(theta)),-1)
        self.x[...,3:].copy_(unit(mul(exp,self.x[...,3:])))
        if self.budget_enabled:self.finish_budget(before,old_x,old_v,vnew,ext,elastic,K,M,pm)

    def _call_fused(self,steps:int):
        self._fused_ext.advance_steps(
            int(steps),int(self.count),float(self.dt),bool(self.ground_contact),float(self.stop_gap),
            bool(self.tendon_crank_mode),float(self.horn_radius),float(self.guide_dist),
            float(self.tendon_slack),float(self.tendon_stiffness),float(self.tendon_damping),
            self.x,self.v,self.servo,self.target,self.yaw,
            self.spin,self.angle,self.memory,self.tension,self.support,
            self.last_slip,self.last_yaw_torque,self.last_yaw_torque_cap,self.failed,
            self.wheel_mu.reshape(-1),self.mass_scale,
            self.yaw_kp,self.yaw_kd,self.yaw_torque_limit,self.yaw_speed_limit,
            self.servo_torque_limit,self.servo_speed_limit,
            self._nodes_i32,self._lengths_c,self._transforms_c,self._rest_local_c,
            self._K_pack,self._M_pack,self._A_pack,
            self._mounts_c,self._anchors_c,self._guides_c,
            self.connector_arm,self.plate_inertia_diag,
            self._plate_ext,self._strip_buf,
            int(self.contact_model_id),bool(self.implicit_contact),
            float(self.ground_kn),float(self.ground_cn),
            float(self.ground_hc_k),float(self.ground_hc_alpha),
            float(self.ground_soil_exp),float(self.ground_soil_unload_ratio),
            float(self.ground_rut_relax_m),float(self.ground_rut_lat_k),
            float(self.wheel_mu_lat_ratio),float(self.wheel_rolling),
            float(self.wheel_patch_radius),float(self.wheel_pivot_mu),
            float(self.ratchet_backlash),
            self.ratchet_play,self.soil_sinkage,self._plate_K_contact,
            self.env_terrain,self.wall_cfg,self.last_wall_force,
            float(self.rayleigh_alpha),self.env_dynamics,self.plate_masses,
            self.rayleigh_beta_env,self.ground_geom,
            self.pipe_cfg,bool(self.strip_node_contact),
            self.last_strip_wall_force,self.last_strip_min_z,
            float(self.wheel_mass_kg),
        )

    def capture(self,steps=1):
        if self.device.type!='cuda':raise ValueError('CUDA graph requires CUDA')
        if self.budget_enabled:raise ValueError('Budget validation uses uncaptured steps; default training remains captured')
        steps=int(steps)
        if self.tangent_interval>1 and steps%self.tangent_interval:
            raise ValueError('Captured step count must be a multiple of tangent_interval')
        state_tensors=(self.x,self.v,self.servo,self.spin,self.angle,self.memory,self.ratchet_play,self.soil_sinkage,self.tension,self.support,self.failed)
        saved=[a.clone() for a in state_tensors]
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            if self._fused_ext is not None:
                for _ in range(3):self._call_fused(steps)
            else:
                for k in range(3*steps):self.physics(reuse_tangent=(self.tangent_interval>1 and k%self.tangent_interval!=0))
        torch.cuda.current_stream().wait_stream(stream)
        self._capture_has_gpipe=bool((self.pipe_cfg[:,0]>0.5).any())
        self._capture_has_legacy_wall=bool((self.pipe_cfg[:,0]<=0.5).any())
        self._capturing_graph=True
        try:
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                if self._fused_ext is not None:
                    self._call_fused(steps)
                else:
                    for k in range(steps):self.physics(reuse_tangent=(self.tangent_interval>1 and k%self.tangent_interval!=0))
        finally:
            self._capturing_graph=False
        for a,b in zip(state_tensors,saved):a.copy_(b)
        self.graph_steps=steps
        self._substep_counter=0

    def advance_steps(self,steps):
        steps=int(steps)
        if self.graph is not None and steps%self.graph_steps==0:
            for _ in range(steps//self.graph_steps):self.graph.replay()
            return
        if self._fused_ext is not None:
            self._call_fused(steps)
            return
        if self.graph is not None:
            raise ValueError('Step count must divide graph length')
        for _ in range(steps):
            reuse=(self.tangent_interval>1 and (self._substep_counter%self.tangent_interval!=0))
            self.physics(reuse_tangent=reuse)
            self._substep_counter+=1
