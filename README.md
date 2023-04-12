# test repo
This particular repo is a copy of the REDQ source code, but modified for other use

NOTE: this is not the official code repo for REDQ paper. Instead see here: https://github.com/watchernyu/REDQ

---

## nyu shanghai hpc singularity setup
Work in progress...

### First time setup: 
```
mkdir /scratch/$USER/.sing_cache
export SINGULARITY_CACHEDIR=/scratch/$USER/.sing_cache
echo "export SINGULARITY_CACHEDIR=/scratch/$USER/.sing_cache" >> ~/.bashrc
mkdir /scratch/$USER/sing
cd /scratch/$USER/sing 
```
### start interactive test session: 
greene cpu session: 
```
srun --pty --cpus-per-task=1 --mem 12000 -t 0-06:00 bash
```
greene gpu session: 
```
srun --pty --gres=gpu:1 --cpus-per-task=4 --mem 12000 -t 0-06:00 bash
```

Shanghai interactive cpu session: 
```
srun -p aquila,parallel --pty --mem 12000 -t 0-05:00 bash
```
Shanghai interactive gpu session: 
```
srun -p aquila,parallel --pty --gres=gpu:1 --mem 12000 -t 0-05:00 bash
```


### Set up d4rl sandbox
```
module load singularity # not needed on greene
cd /scratch/$USER/sing/
singularity build --sandbox d4rl-sandbox docker://cwatcherw/d4rl:0.1
```

### Run
```
cd /scratch/$USER/sing
singularity exec --nv -B /scratch/$USER/sing/combination/:/code -B /scratch/$USER/sing/d4rl-sandbox/opt/conda/lib/python3.8/site-packages/mujoco_py/:/opt/conda/lib/python3.8/site-packages/mujoco_py/ /scratch/$USER/sing/d4rl-sandbox bash
```

### env variables
```
export PYTHONPATH=$PYTHONPATH:/code
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=/workspace/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL=egl
cd /code
```

### zip and send back data
```
zip -r ../sendback/il.zip il_*
```








---


## Combination alg doc below:

Run combination alg: 

```
singularity exec --nv -B /scratch/$USER/sing/combination:/code -B /scratch/$USER/sing/vrl3sing/opt/conda/lib/python3.8/site-packages/mujoco_py/:/opt/conda/lib/python3.8/site-packages/mujoco_py/ -B /scratch/$USER/sing/combdata:/combdata /scratch/$USER/sing/vrl3sing bash

export PYTHONPATH=$PYTHONPATH:/code
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=/workspace/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL=egl
```


old: 
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=/workspace/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib
export MUJOCO_GL=egl
cd /workspace/REDQ/experiments
python proj1.py 
```
