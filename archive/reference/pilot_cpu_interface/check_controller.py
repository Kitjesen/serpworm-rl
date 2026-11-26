"""Independent ODE integration, coordinate, and no-CPG observation checks."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from matsuoka import Matsuoka


def main():
    records=[]; traces={}
    for c in (0., .75):
        for name, alpha in [('symmetric',np.zeros((1,4))),('biased',np.full((1,4),.6))]:
            model=Matsuoka(c=c); initial=model.state.copy()
            exact=solve_ivp(lambda t,s:model.rhs(s.reshape(initial.shape),alpha).ravel(),[0,8],initial.ravel(),rtol=1e-9,atol=1e-11,max_step=.005)
            t=np.arange(400)*.02; output=np.array([model.step(alpha)[0] for _ in t])
            error=float(np.max(abs(model.state.ravel()-exact.y[:,-1])))
            assert error<1e-5,(c,name,error)
            fine=Matsuoka(c=c,h=.00125)
            finer=np.array([fine.step(alpha)[0] for _ in t])
            output_error=float(np.max(abs(output-finer)))
            assert output_error<1e-4
            records.append(dict(c=c,input=name,rk4_vs_scipy_state_error=error,half_step_output_error=output_error,output_peak=np.max(abs(output),axis=0).tolist(),last_4s_peak_to_peak=np.ptp(output[200:],axis=0).tolist()))
            traces[f'c{c}_{name}']=output
    model=Matsuoka(c=.75)
    initial=model.state.copy();t_long=np.arange(0,60,.02)
    solution=solve_ivp(lambda t,s:model.rhs(s.reshape(initial.shape),np.zeros((1,4))).ravel(),[0,60],initial.ravel(),rtol=1e-8,atol=1e-10,t_eval=t_long)
    z=np.maximum(solution.y.T.reshape(-1,4,2,2)[...,0],0)
    output_long=2.0935*(z[...,0]-z[...,1])
    assert np.all(np.ptp(output_long[-1000:],axis=0)>.5),'FOC free response lost oscillation'
    assert model.w[2,1]==8.8669 and model.w[0,2]==.7844 and np.count_nonzero(model.w)==12
    np.savez_compressed(Path(__file__).with_name('foc_free_response.npz'),time_s=t_long,output=output_long)
    # The selected controller frame is plate4 local frame rotated 180deg about Z.
    basis=np.diag([-1.,-1.,1.])
    assert np.linalg.det(basis)==1
    assert np.allclose(basis.T@[-.01,0,0],[.01,0,0])
    assert np.allclose(basis.T@[0,-.01,0],[0,.01,0])
    np.savez_compressed(Path(__file__).with_name('controller_traces.npz'),time_s=t,**traces)
    Path(__file__).with_name('controller_checks.json').write_text(json.dumps(dict(checks=records,coordinate='right handed, forward -worldX, left -worldY, up +worldZ',scope='ODE integration verified; author code equivalence, locomotion and physical feasibility not established'),indent=2))
    print(json.dumps(records),flush=True)


if __name__=='__main__': main()
