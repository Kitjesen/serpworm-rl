"""Two finite, matched long runs. New reward first; no incumbent overwrite."""
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
if __name__ == '__main__':
    for reward in ('tracking_v2','legacy'):
        out=root/'results'/('long_'+reward+'_seed7')
        command=[sys.executable,'-u',str(root/'train_gpu.py'),'--kind','direct',
                 '--reward',reward,'--envs','8','--horizon','64','--rollouts','256',
                 '--seconds','0','--seed','7','--save-every','32','--eval-every','32','--out',str(out)]
        with (root/('long_'+reward+'.log')).open('w') as log:
            subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
