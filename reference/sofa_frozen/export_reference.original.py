"""Export frozen SOFA topology and one-step force/matrix oracles; offline only."""
import sys, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'v1'/'source'))
from sofa_worm_env import SofaWormEnv, quat
import Sofa
from Sofa import SofaLinearSystem
import scipy.sparse

if __name__=='__main__':
    for count in (1,2,5):
        e=SofaWormEnv(count=count,elements=16,randomize=False,dt=.00025,beam_model='saint_venant',rolling=.06)
        e.factors=np.ones(4);e.build()
        beam=e.root.getObject('Beams')
        ff=next(o for o in e.root.objects if o.getClassName()=='AdaptiveBeamForceFieldAndMass')
        print(count,[(o.getName(),o.getClassName()) for o in e.root.objects],flush=True)
        print('methods',[x for x in dir(ff) if 'atrix' in x or 'orce' in x],flush=True)
        transforms=np.zeros((8*count,16,2,7));transforms[...,6]=1
        for strip in range(8*count):
            k=strip%8
            front=np.mean(e.params['front_clamp_hole_pairs_m'][k],axis=0)
            back=np.mean(e.params['back_clamp_hole_pairs_m'][k],axis=0)
            radial=np.array([0.,front[1],front[2]]);radial/=np.linalg.norm(radial)
            side=np.subtract(*e.params['front_clamp_hole_pairs_m'][k]);side/=np.linalg.norm(side)
            a=e.rest[2*(strip//8),:3]+front;b=e.rest[2*(strip//8)+1,:3]+back
            for edge,endpoint,u,mount in [(0,0,0.,front),(15,1,1.,back)]:
                tangent=b-a+4*e.params['stress_free_bow_m']*(1-2*u)*radial;tangent/=np.linalg.norm(tangent)
                width=side-np.dot(side,tangent)*tangent;width/=np.linalg.norm(width)
                transforms[strip,edge,endpoint]=np.r_[mount,quat(np.column_stack([tangent,width,np.cross(tangent,width)]))]
        data=dict(rest=e.rest,edges=e.root.getObject('Edges').edges.array(),lengths=beam.lengthList.array(),
                  transforms=transforms,
                  strip_nodes=e.strip_nodes,wheel_mounts=e.wheel_mounts,anchors=e.anchors,guides=e.guides)
        out=ROOT/'inputs';out.mkdir(exist_ok=True)
        np.savez(out/f'topology_{count}.npz',**data)
        (out/'parameters.json').write_text(json.dumps(e.params,indent=2))
        e.reset(seed=7)
        e.servo[:]=.3
        before=dict(x=e.dofs.position.array().copy(),v=e.dofs.velocity.array().copy(),servo=e.servo.copy(),spin=e.wheel_speed.copy(),angle=e.wheel_angle.copy(),memory=e.contact_memory.copy(),tension=e.tensions.copy())
        original=Sofa.Simulation.animate
        external=[]
        def capture(root,dt):
            external.append(e.dofs.externalForce.array().copy())
            original(root,dt)
        Sofa.Simulation.animate=capture
        e.physics()
        Sofa.Simulation.animate=original
        system=e.root.getObject('MatrixLinearSystem1')
        scipy.sparse.save_npz(out/f'matrix_{count}.npz',system.get_system_matrix())
        np.save(out/f'rhs_{count}.npy',system.get_rhs_vector())
        print('system methods',[(o.getName(),[m for m in dir(o) if 'atrix' in m or 'ector' in m]) for o in e.root.objects if 'Solver' in o.getClassName() or 'System' in o.getClassName()],flush=True)
        np.savez(out/f'oracle_{count}.npz',**before,next_x=e.dofs.position.array().copy(),next_v=e.dofs.velocity.array().copy(),external=external[0],force=e.dofs.force.array(),next_tension=e.tensions.copy(),next_spin=e.wheel_speed.copy())
        e.close()
