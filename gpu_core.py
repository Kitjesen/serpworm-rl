"""Device-resident corotational beam/rigid/tendon/wheel reference implementation.

Beam matrices and spline frame follow BeamAdapter (SOFA, LGPL-2.1-or-later),
whose source evidence and license are archived in reference/BeamAdapter.
No SOFA calls, NumPy mechanics, or device-to-host reads inside physics().
Explicit coupling is intentionally retained for comparison with frozen v1.
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from wheel_stiction import contact as implicit_wheel_contact

ROOT=Path(__file__).resolve().parent

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
    # Near reversal is outside this validated small-element-deformation domain.
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
    """Analytic du/d(world endpoint twist), including the spline-frame motion.

    This is a strain derivative, separate from the physical twist map used by M.
    Compression follows midpoint's branch; branch boundaries and near reversal
    are outside the smooth small-element-deformation validation domain.
    """
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

def local_matrices(lengths,p):
    """Cold setup only; exact section formulas used by the pinned BeamAdapter."""
    out=[];w=p['strip_width_m'];t=p['strip_thickness_m'];A=w*t
    Iy=w*t**3/12;Iz=t*w**3/12;J=w*t**3*(1/3-.21*t/w*(1-t**4/(12*w**4)))
    E=p['youngs_modulus_pa'];G=E/(2*(1+p['poisson_ratio']));rho=p['steel_density_kg_m3']
    for L in lengths.ravel():
        k=np.zeros((12,12));m=np.zeros((12,12));a=k[:6,:6];b=k[6:,6:];c=k[6:,:6]
        d=[E*A/L,12*E*Iz/L**3,12*E*Iy/L**3,G*J/L,4*E*Iy/L,4*E*Iz/L]
        a[np.diag_indices(6)]=d;b[np.diag_indices(6)]=d
        a[4,2]=-6*E*Iy/L**2;a[5,1]=6*E*Iz/L**2;b[4,2]=-a[4,2];b[5,1]=-a[5,1]
        c[np.diag_indices(6)]=[-d[0],-d[1],-d[2],-d[3],2*E*Iy/L,2*E*Iz/L]
        c[1,5]=-a[5,1];c[2,4]=-a[4,2];c[4,2]=a[4,2];c[5,1]=a[5,1]
        a[:]=a+np.tril(a,-1).T;b[:]=b+np.tril(b,-1).T;k[:6,6:]=c.T
        a=m[:6,:6];b=m[6:,6:];c=m[6:,:6]
        d=[1/3,13/35+6*Iz/(5*A*L**2),13/35+6*Iy/(5*A*L**2),(Iy+Iz)/(3*A),L**2/105+2*Iy/(15*A),L**2/105+2*Iz/(15*A)]
        a[np.diag_indices(6)]=d;b[np.diag_indices(6)]=d
        a[4,2]=-11*L/210-Iy/(10*A*L);a[5,1]=11*L/210+Iz/(10*A*L);b[4,2]=-a[4,2];b[5,1]=-a[5,1]
        c[np.diag_indices(6)]=[1/6,9/70-6*Iz/(5*A*L**2),9/70-6*Iy/(5*A*L**2),(Iy+Iz)/(6*A),-L**2/140-Iy/(30*A),-L**2/140-Iz/(30*A)]
        c[1,5]=13*L/420-Iz/(10*A*L);c[2,4]=-13*L/420+Iy/(10*A*L);c[4,2]=-c[2,4];c[5,1]=-c[1,5]
        a[:]=a+np.tril(a,-1).T;b[:]=b+np.tril(b,-1).T;m[:6,6:]=c.T;m*=rho*A*L
        out.append((k,m))
    return np.array([x[0] for x in out]),np.array([x[1] for x in out])

class GPUCore:
    def __init__(self,count=1,batch=1,dt=.00025,device='cuda',dtype=torch.float64,sofa_mass_compat=False,legacy_rolling=False,yaw_motor_speed=None,ground_contact=True,budget=False,strain_jacobian=False,plate_solver='legacy'):
        if count not in (1,2,5) or batch<1 or not 0<dt<=.001:raise ValueError('Invalid topology/time step')
        if plate_solver not in ('legacy','segment_lu'):raise ValueError('Invalid plate solver')
        self.count,self.batch,self.dt,self.device,self.dtype=count,batch,dt,torch.device(device),dtype
        self.plate_solver=plate_solver
        self.sofa_mass_compat=sofa_mass_compat
        self.legacy_rolling=legacy_rolling or sofa_mass_compat
        self.yaw_motor_speed=yaw_motor_speed
        self.ground_contact=bool(ground_contact)
        self.budget_enabled=bool(budget)
        self.strain_jacobian=bool(strain_jacobian)
        self.last_budget=None
        self.wheel_mu=torch.full((batch,1,1),.8,device=self.device,dtype=dtype)
        raw=np.load(ROOT/'inputs'/f'topology_{count}.npz');self.parameters=json.loads((ROOT/'inputs'/'parameters.json').read_text())
        self.nodes=self.tensor(raw['strip_nodes'],torch.long);self.edges=self.tensor(raw['edges'],torch.long)
        self.S,self.N=self.nodes.shape;self.E=self.N-1;self.P=2*count;self.D=len(raw['rest'])
        if plate_solver=='segment_lu':
            # Cold CPU check: each strip belongs to its ordered pair of plates.
            expected=np.repeat(np.arange(self.P).reshape(count,2),self.S//count,axis=0)
            if self.S%count or not np.array_equal(raw['strip_nodes'][:,[0,-1]],expected):
                raise ValueError('segment_lu requires strips grouped by adjacent segment plate pairs')
            self.segment_ids=torch.arange(count,device=self.device)
        self.lengths=self.tensor(raw['lengths'].reshape(self.S,self.E))
        self.transforms=self.tensor(raw['transforms']);self.mounts=self.tensor(raw['wheel_mounts'])[torch.arange(self.P,device=self.device)%2]
        self.anchors=self.tensor(raw['anchors']);self.guides=self.tensor(raw['guides'])
        self.x=self.tensor(raw['rest'])[None].repeat(batch,1,1);self.rest=self.x.clone();self.v=torch.zeros((batch,self.D,6),device=self.device,dtype=dtype)
        self.servo=torch.zeros((batch,2*count),device=self.device,dtype=dtype);self.target=self.servo.clone();self.yaw=torch.zeros((batch,count-1),device=self.device,dtype=dtype)
        self.spin=torch.zeros((batch,self.P,2),device=self.device,dtype=dtype);self.angle=self.spin.clone();self.memory=torch.zeros((batch,self.P,2,3),device=self.device,dtype=dtype)
        self.tension=torch.zeros((batch,count,4),device=self.device,dtype=dtype);self.support=self.spin.clone()
        k,m=local_matrices(raw['lengths'],self.parameters);self.K=self.tensor(k).reshape(self.S,self.E,12,12);self.M=self.tensor(m).reshape(self.S,self.E,12,12)
        self.rest_local=self.frames()[0]
        self.endpoint_ids=self.nodes[:,[0,-1]].reshape(-1)
        self.interior_ids=self.nodes[:,1:-1].reshape(-1)
        self.plate_dofs=(self.nodes[:,[0,-1],None]*6+torch.arange(6,device=self.device)).reshape(self.S,12)
        self.scatter_mat=(self.plate_dofs[:,:,None]*(self.P*6)+self.plate_dofs[:,None,:]).reshape(-1)
        self.gravity=torch.zeros_like(self.v);self.gravity[...,2]=-9.81
        self.connector_arm=self.tensor([-.058,0.,-.0010452742534])
        self.plate_inertia=torch.diag(self.tensor([.0016,.0009,.0009])*.28)
        self.failed=torch.zeros((),device=self.device,dtype=torch.bool)
        self.graph=None

    def tensor(self,x,dtype=None):return torch.as_tensor(x,device=self.device,dtype=dtype or self.dtype)

    def frames(self,with_jacobian=False):
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
        return u,T

    def beam(self):
        if self.strain_jacobian:
            u,T,J=self.frames(with_jacobian=True)
            elastic=-(J.transpose(-1,-2)@(self.K@(u-self.rest_local)[...,None]))[...,0]
            # Material quasi-Newton tangent: stress-dependent geometric Hessian
            # terms remain omitted. Physical twist transport T still governs M.
            K=J.transpose(-1,-2)@self.K@J;M=T.transpose(-1,-2)@self.M@T
            return self.scatter_vector(elastic),K,M
        u,T=self.frames();elastic=-(T.transpose(-1,-2)@(self.K@(u-self.rest_local)[...,None]))[...,0]
        K=T.transpose(-1,-2)@self.K@T;M=T.transpose(-1,-2)@self.M@T
        f=torch.zeros_like(self.v)
        f.index_add_(1,self.edges.reshape(-1),elastic.reshape(self.batch,-1,6))
        return f,K,M

    def external(self):
        h=self.dt;x=self.x[:,:self.P];v=self.v[:,:self.P];q=x[...,3:];R=matrix(q);f=torch.zeros_like(self.v)
        # Previous-step load changes motor speed, as in the CPU oracle.
        radius=.0235;guide=.035;alpha=math.acos(radius/guide)
        chord=torch.sqrt(guide**2+radius**2-2*guide*radius*torch.cos(self.servo))
        lever=torch.where(self.servo<=alpha,radius*guide*torch.sin(self.servo)/chord,radius)
        load=self.tension.reshape(self.batch,2*self.count,2).sum(-1)*lever
        inc=(self.target-self.servo).clamp(-2*h,2*h);inc=inc*torch.where(inc>0,(1-load/.5).clamp(0,1),1.)
        self.servo.add_(inc)
        chord=torch.sqrt(guide**2+radius**2-2*guide*radius*torch.cos(self.servo))
        takeup=torch.where(self.servo<=alpha,chord,math.sqrt(guide**2-radius**2)+radius*(self.servo-alpha))-(guide-radius)
        ar=rotate(q[:,0::2,None,:].expand(-1,-1,4,-1),self.anchors.expand(self.batch,self.count,-1,-1))
        br=rotate(q[:,1::2,None,:].expand(-1,-1,4,-1),self.guides.expand(self.batch,self.count,-1,-1))
        delta=x[:,1::2,None,:3]+br-x[:,0::2,None,:3]-ar;length=norm(delta);direction=delta/length.clamp_min(1e-15)
        va=v[:,0::2,None,:3]+cross(v[:,0::2,None,3:].expand_as(ar),ar);vb=v[:,1::2,None,:3]+cross(v[:,1::2,None,3:].expand_as(br),br)
        rest=.101-takeup.reshape(self.batch,self.count,2).repeat_interleave(2,-1)
        extension=length[...,0]-rest;speed=((vb-va)*direction).sum(-1)
        self.tension.copy_(torch.where(extension>=0,(2000*extension+.5*speed).clamp_min(0),0.))
        force=self.tension[...,None]*direction
        f[:,:self.P:2,:3]+=force.sum(-2);f[:,1:self.P:2,:3]-=force.sum(-2)
        f[:,:self.P:2,3:]+=cross(ar,force).sum(-2);f[:,1:self.P:2,3:]-=cross(br,force).sum(-2)
        sep=x[:,1::2,:3]-x[:,0::2,:3];axis=-R[:,0::2,:,0];gap=(sep*axis).sum(-1)
        push=50000*(.075-gap).clamp_min(0)[...,None]*axis
        f[:,1:self.P:2,:3]+=push;f[:,:self.P:2,:3]-=push;f[:,:self.P:2,3:]-=cross(sep,push)
        if self.count>1:
            a=slice(1,self.P-1,2);b=slice(2,self.P,2)
            arm=rotate(q[:,a],self.connector_arm.expand(self.batch,self.count-1,-1))
            sep=x[:,b,:3]-x[:,a,:3];err=sep-arm
            relv=v[:,b,:3]-v[:,a,:3]-cross(v[:,a,3:],sep);force=-10000*err-10*relv
            f[:,b,:3]+=force;f[:,a,:3]-=force;f[:,a,3:]-=cross(sep,force)
            relative=R[:,a].transpose(-1,-2)@R[:,b];yaw=torch.atan2(relative[...,1,0],relative[...,0,0]);axis=R[:,a,:,2]
            omega=v[:,b,3:]-v[:,a,3:];rate=(omega*axis).sum(-1,keepdim=True)
            yaw_torque=(2*(self.yaw-yaw)[...,None]-.03*rate).clamp(-.5,.5)
            if self.yaw_motor_speed is not None:
                cap=.5*torch.where(yaw_torque*rate>0,(1-rate.abs()/self.yaw_motor_speed).clamp(0,1),1.)
                yaw_torque=torch.maximum(torch.minimum(yaw_torque,cap),-cap)
                self.last_yaw_torque_cap=cap[...,0]
            else:self.last_yaw_torque_cap=torch.full_like(yaw,.5)
            self.last_yaw_torque=yaw_torque[...,0]
            torque=yaw_torque*axis+20*cross(R[:,b,:,2],axis)-.1*(omega-axis*rate)
            f[:,b,3:]+=torque;f[:,a,3:]-=torque
        self.internal_force=f.sum(1)[:,:3]
        self.internal_torque=(cross(self.x[...,:3],f[...,:3])+f[...,3:]).sum(1)
        if self.budget_enabled:self.last_internal_load=f.clone()
        # Same unilateral wheel model, vectorized over every wheel/environment.
        axle=R[...,1];up=torch.zeros_like(axle);up[...,2]=1
        forward=unit(cross(up,axle));lateral=cross(up,forward);rim=-.018*unit(up-axle*axle[...,2:])
        offset=rotate(q[...,None,:].expand(-1,-1,2,-1),self.mounts.expand(self.batch,-1,-1,-1))
        velocity=v[...,None,:3]+cross(v[...,None,3:].expand_as(offset),offset+rim[...,None,:])
        penetration=-(x[...,None,2]+offset[...,2]+rim[...,None,2]);active=(penetration>0)&self.ground_contact
        normal=torch.where(active,(5000*penetration-8*velocity[...,2]).clamp_min(0),0.)
        basis=torch.stack((forward,lateral),-2)
        tangent_v=torch.einsum('bpkc,bpsc->bpsk',basis,velocity);mem=torch.einsum('bpkc,bpsc->bpsk',basis,self.memory)
        inertia=.5*.009*.018**2;gain=2000*h+8
        resistance=torch.minimum(.06*normal*.018+2e-6*self.spin,inertia*self.spin/h)
        base=-2000*mem-gain*tangent_v
        omega=((self.spin-h*(.018*base[...,0]+resistance)/inertia)/(1+h*.018**2*gain/inertia)).clamp_min(0)
        tang=base.clone();tang[...,0]+=gain*.018*omega
        mag=norm(tang);cap=.8*normal;sliding=mag[...,0]>cap
        tang=tang*torch.minimum(torch.ones_like(mag),cap[...,None]/mag.clamp_min(1e-15))
        free=self.spin-h*(.018*tang[...,0]+resistance)/inertia;omega=free.clamp_min(0)
        reaction=inertia*(omega-free)/h-resistance
        slip=tangent_v.clone();slip[...,0]-=.018*omega
        mem=torch.where(sliding[...,None],-(tang+8*slip)/2000,mem+h*slip)
        if not self.legacy_rolling:
            original_mem=torch.einsum('bpkc,bpsc->bpsk',basis,self.memory)
            tang,omega,mem,reaction,slip=implicit_wheel_contact(tangent_v,self.spin,original_mem,normal,h,mu=self.wheel_mu)
        self.last_slip=slip
        self.memory.copy_(torch.where(active[...,None],torch.einsum('bpsk,bpkc->bpsc',mem,basis),0.))
        self.spin.copy_(torch.where(active,omega,self.spin*max(0,1-h*2e-6/inertia)));self.angle.add_(h*self.spin);self.support.copy_(normal)
        force=torch.einsum('bpsk,bpkc->bpsc',tang,basis);force[...,2]+=normal;force=torch.where(active[...,None],force,0.)
        rt=cross(rim[...,None,:].expand_as(force),force)
        torque=cross(offset,force)+rt-axle[...,None,:]*(rt*axle[...,None,:]).sum(-1,keepdim=True)+torch.where(active,reaction,0.)[...,None]*axle[...,None,:]
        f[:,:self.P,:3]+=force.sum(-2);f[:,:self.P,3:]+=torque.sum(-2)
        if self.budget_enabled:
            self.last_ground_load=f-self.last_internal_load
            self.last_ground_force=force.clone()
            self.last_contact_position=x[...,None,:3]+offset+rim[...,None,:]
        return f

    def mass_action(self,velocity,M=None,pm=None):
        """All-node generalized momentum using consistent beam and plate mass."""
        if M is None:_,_,M=self.beam()
        if pm is None:pm=self.plate_mass()
        ve=velocity[:,self.edges].reshape(self.batch,self.S,self.E,12)
        p=self.scatter_vector((M@ve[...,None])[...,0])
        p[:,:self.P]+=(pm@velocity[:,:self.P,...,None])[...,0]
        return p

    def project_resultant(self,load,position=None):
        """SE(3) virtual-work projection; includes every beam and plate node."""
        if position is None:position=self.x[...,:3]
        return torch.cat((load[...,:3].sum(1),
            (cross(position,load[...,:3])+load[...,3:]).sum(1)),-1)

    def budget_state(self):
        """Read-only candidate energy/momentum measures, outside CUDA capture.

        Beam energy uses the co-rotational local strain quadratic. It is not a
        proof that the frozen-frame force/tangent is its exact global derivative.
        Wheel translation is within the unmeasured .28 kg plate lump; only the
        relative wheel spin term is added separately, with positive spin -axle.
        """
        u,T=self.frames();M=T.transpose(-1,-2)@self.M@T
        pm=self.plate_mass();p=self.mass_action(self.v,M,pm)
        du=u-self.rest_local
        kinetic=.5*(self.v*p).sum((1,2))
        beam=.5*(du*(self.K@du[...,None])[...,0]).sum((1,2,3))
        R=matrix(self.x[:,:self.P,3:]);axle=R[...,1]
        wheel_inertia=.5*.009*.018**2
        spin_momentum=-(wheel_inertia*self.spin[...,None]*axle[...,None,:]).sum((1,2))
        spin_energy=.5*wheel_inertia*(self.spin**2).sum((1,2))
        ar=rotate(self.x[:,:self.P:2,None,3:].expand(-1,-1,4,-1),self.anchors.expand(self.batch,self.count,-1,-1))
        br=rotate(self.x[:,1:self.P:2,None,3:].expand(-1,-1,4,-1),self.guides.expand(self.batch,self.count,-1,-1))
        delta=self.x[:,1:self.P:2,None,:3]+br-self.x[:,:self.P:2,None,:3]-ar
        radius,guide=.0235,.035;alpha=math.acos(radius/guide)
        chord=torch.sqrt(guide**2+radius**2-2*guide*radius*torch.cos(self.servo))
        takeup=torch.where(self.servo<=alpha,chord,math.sqrt(guide**2-radius**2)+radius*(self.servo-alpha))-(guide-radius)
        extension=norm(delta)[...,0]-.101+takeup.reshape(self.batch,self.count,2).repeat_interleave(2,-1)
        tendon=1000*extension.clamp_min(0).square().sum((1,2))
        gap=((self.x[:,1:self.P:2,:3]-self.x[:,:self.P:2,:3])*(-R[:,:self.P:2,:,0])).sum(-1)
        stop=25000*(.075-gap).clamp_min(0).square().sum(1)
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
        normal_storage=2500*penetration.clamp_min(0).square().sum((1,2))*self.ground_contact
        tangential_storage=1000*self.memory.square().sum((1,2,3))*self.ground_contact
        # The normal/tangential springs are regularized contact storage; clipping
        # and airborne memory resets can discard energy and must remain visible.
        gravity_force=self.mass_action(self.gravity,M,pm)
        gravity_potential=-(self.x[...,:3]*gravity_force[...,:3]).sum((1,2))
        resultant=self.project_resultant(p)
        energies=dict(kinetic=kinetic,beam=beam,tendon=tendon,stop=stop,
            connector=connector,gravity=gravity_potential,wheel_spin=spin_energy,
            contact_normal=normal_storage,contact_tangent=tangential_storage)
        return dict(momentum=resultant,spin_momentum=spin_momentum,
            energy=energies,total_energy=sum(energies.values()),takeup=takeup)

    def finish_budget(self,before,old_x,old_v,vnew,ext,elastic,K,M,pm):
        """Expose discrete closure and frame-transport residuals, not certificates."""
        h=self.dt;after=self.budget_state()
        p0=self.mass_action(old_v,M,pm);p1=self.mass_action(vnew,M,pm)
        ve=vnew[:,self.edges].reshape(self.batch,self.S,self.E,12)
        kv=self.scatter_vector((K@ve[...,None])[...,0])
        gravity=self.mass_action(self.gravity,M,pm)
        rayleigh=-.05*p1-.002*kv
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
        # Ground contact torque is evaluated at the physical rim, so adding the
        # -axle wheel spin removes the carrier's internal axle reaction torque.
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
        R=matrix(self.x[:,:self.P,3:]);m=torch.zeros((self.batch,self.P,6,6),device=self.device,dtype=self.dtype)
        m[...,:3,:3]=torch.eye(3,device=self.device,dtype=self.dtype)*.28
        # Preserve v1 UniformMass's constant inertia matrix for migration parity.
        m[...,3:,3:]=self.plate_inertia
        return m

    def solve(self,A,rhs,plate):
        """Batched block Thomas elimination, then a 6*(2*count) plate Schur solve."""
        diag=A[:,:,:-1,6:,6:]+A[:,:,1:,:6,:6];lower=A[:,:,1:-1,6:,:6];upper=A[:,:,1:-1,:6,6:]
        r=torch.zeros((self.batch,self.S,self.E-1,6,13),device=self.device,dtype=self.dtype)
        r[...,0]=rhs[:,self.nodes[:,1:-1]]
        r[:,:,0,:,1:7]=A[:,:,0,6:,:6];r[:,:,-1,:,7:]=A[:,:,-1,:6,6:]
        ds=[];rs=[];status=[]
        for i in range(self.E-1):
            d=diag[:,:,i];b=r[:,:,i]
            if i:
                factor=lower[:,:,i-1]@ds[-1];d=d-factor@upper[:,:,i-1];b=b-factor@rs[-1]
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
            # Valid for the current split coupling: inter-segment forces are in
            # rhs only. Revisit this decomposition if implicit joint/contact
            # terms introduce off-diagonal segment blocks in globalA.
            blocks=globalA.reshape(self.batch,self.count,12,self.count,12).permute(0,1,3,2,4)
            local=blocks[:,self.segment_ids,self.segment_ids].reshape(self.batch*self.count,12,12)
            vp,info=torch.linalg.solve_ex(local,globalr.reshape(self.batch*self.count,12,1),check_errors=False)
            vp=vp.reshape(self.batch,P,1)
        elif self.batch<=8:
            vp,info=torch.linalg.solve_ex(globalA,globalr[...,None],check_errors=False)
        else:
            # Torch's larger-batch dispatch uses a solver path which synchronizes
            # during graph capture. Small GPU-only chunks retain capture support.
            chunks=[torch.linalg.solve_ex(globalA[i:i+8],globalr[i:i+8,:,None],check_errors=False) for i in range(0,self.batch,8)]
            vp=torch.cat([s.result for s in chunks],0);info=torch.cat([s.info for s in chunks],0)
        vp=vp[...,0]
        self.failed.logical_or_(torch.stack(status).any()|(info!=0).any())
        vi=z[...,0]-(z[...,1:]@vp[:,self.plate_dofs][...,None,:,None])[...,0]
        out=torch.empty_like(rhs);out[:,:self.P]=vp.reshape(self.batch,self.P,6);out[:,self.interior_ids]=vi.reshape(self.batch,-1,6)
        return out

    def physics(self):
        if self.budget_enabled:
            before=self.budget_state();old_x=self.x[...,:3].clone();old_v=self.v.clone()
        h=self.dt;ext=self.external();elastic,K,M=self.beam();pm=self.plate_mass()
        vg=(self.v+h*self.gravity)[:,self.edges].reshape(self.batch,self.S,self.E,12)
        rhs=self.scatter_vector((M@vg[...,None])[...,0])+h*(ext+elastic)
        if self.sofa_mass_compat:
            # Oracle-only reproduction: installed BeamAdapter addMDx ignores dx
            # and multiplies gravity. Its Rayleigh-mass RHS is consequently wrong.
            diff=(self.v-self.gravity)[:,self.edges].reshape(self.batch,self.S,self.E,12)
            rhs+=h*.05*self.scatter_vector((M@diff[...,None])[...,0])
        rhs[:,:self.P]+=(pm@(self.v[:,:self.P]+h*self.gravity[:,:self.P])[...,None])[...,0]
        A=(1+h*.05)*M+(h*.002+h*h)*K
        vnew=self.solve(A,rhs,(1+h*.05)*pm)
        self.v.copy_(vnew);self.x[...,:3].add_(h*vnew[...,:3])
        # Match installed SOFA RigidCoord's R3 x SO(3) exponential update.
        omega=.5*h*vnew[...,3:];theta=norm(omega)
        exp=torch.cat((omega*torch.sinc(theta/math.pi),torch.cos(theta)),-1)
        self.x[...,3:].copy_(unit(mul(exp,self.x[...,3:])))
        if self.budget_enabled:self.finish_budget(before,old_x,old_v,vnew,ext,elastic,K,M,pm)

    def capture(self,steps=1):
        if self.device.type!='cuda':raise ValueError('CUDA graph requires CUDA')
        if self.budget_enabled:raise ValueError('Budget validation uses uncaptured steps; default training remains captured')
        state=[a.clone() for a in (self.x,self.v,self.servo,self.spin,self.angle,self.memory,self.tension,self.support,self.failed)]
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):self.physics()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            for _ in range(steps):self.physics()
        for a,b in zip((self.x,self.v,self.servo,self.spin,self.angle,self.memory,self.tension,self.support,self.failed),state):a.copy_(b)
        self.graph_steps=steps

    def advance_steps(self,steps):
        if self.graph:
            if steps%self.graph_steps:raise ValueError('Step count must divide graph length')
            for _ in range(steps//self.graph_steps):self.graph.replay()
        else:
            for _ in range(steps):self.physics()
