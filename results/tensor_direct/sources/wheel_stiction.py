"""Rolling torque and wheel spin solved together, including zero-speed stiction."""
import torch

def contact(velocity,spin,memory,normal,dt,radius=.018,mu=.8,rolling=.06):
    inertia=.5*.009*radius**2;stiffness=2000.;damping=8.;viscous=2e-6
    gain=stiffness*dt+damping;base=-stiffness*memory-gain*velocity
    limit=rolling*normal*radius
    omega=((inertia*spin/dt-radius*base[...,0]-limit)/(inertia/dt+radius**2*gain+viscous)).clamp_min(0)
    force=base.clone();force[...,0]+=gain*radius*omega
    magnitude=torch.linalg.vector_norm(force,dim=-1);cap=mu*normal;sliding=magnitude>cap
    force=force*torch.minimum(torch.ones_like(magnitude),cap/magnitude.clamp_min(1e-15))[...,None]
    omega=((inertia*spin/dt-radius*force[...,0]-limit)/(inertia/dt+viscous)).clamp_min(0)
    # Combined axle torque (rolling resistance + one-way clutch). It transmits
    # load even at zero speed, without applying an unbalanced impulsive brake.
    reaction=inertia*(omega-spin)/dt+radius*force[...,0]
    slip=velocity.clone();slip[...,0]-=radius*omega
    memory=torch.where(sliding[...,None],-(force+damping*slip)/stiffness,memory+dt*slip)
    return force,omega,memory,reaction,slip

def old_contact(velocity,spin,memory,normal,dt,radius=.018,mu=.8,rolling=.06):
    inertia=.5*.009*radius**2;gain=2000*dt+8
    resistance=torch.minimum(rolling*normal*radius+2e-6*spin,inertia*spin/dt)
    base=-2000*memory-gain*velocity
    omega=((spin-dt*(radius*base[...,0]+resistance)/inertia)/(1+dt*radius**2*gain/inertia)).clamp_min(0)
    force=base.clone();force[...,0]+=gain*radius*omega
    magnitude=torch.linalg.vector_norm(force,dim=-1);cap=mu*normal;sliding=magnitude>cap
    force*=torch.minimum(torch.ones_like(magnitude),cap/magnitude.clamp_min(1e-15))[...,None]
    free=spin-dt*(radius*force[...,0]+resistance)/inertia;omega=free.clamp_min(0)
    reaction=inertia*(omega-free)/dt-resistance
    slip=velocity.clone();slip[...,0]-=radius*omega
    memory=torch.where(sliding[...,None],-(force+8*slip)/2000,memory+dt*slip)
    return force,omega,memory,reaction,slip

if __name__=='__main__':
    import json
    from pathlib import Path
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    report=[]
    for dt in (.0005,.00025,.000125):
        for name,fn in [('legacy',old_contact),('implicit',contact)]:
            # Under rolling threshold; above threshold; opposing ratchet;
            # contact Coulomb overload; passive coasting.
            drive=torch.tensor([.04,.12,-.04,-2.,0.]);velocity=torch.tensor([0.,0.,0.,0.,.1]);position=torch.zeros(5)
            spin=velocity/.018;memory=torch.zeros((5,2));normal=torch.ones(5)*1.4
            for _ in range(round(1.5/dt)):
                vv=torch.stack((velocity,torch.zeros_like(velocity)),-1)
                force,spin,memory,_,_=fn(vv,spin,memory,normal,dt)
                velocity+=dt*(drive+force[:,0])/.14;position+=dt*velocity
            record=dict(dt=dt,mode=name,velocity_m_s=velocity.tolist(),position_m=position.tolist(),spin_rad_s=spin.tolist())
            if name=='implicit':
                assert abs(velocity[0])<1e-8 and abs(velocity[2])<1e-8,record
                assert abs(position[0]-.04/2000)<1e-6,record
                assert velocity[1]>.01 and velocity[3]<-.1 and abs(velocity[4])<1e-5,record
            report.append(record)
    Path(__file__).with_name('wheel_stiction_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
