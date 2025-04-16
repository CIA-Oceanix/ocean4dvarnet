# Setting up a CUDA + PyTorch environment wtih Mamba


---
## **Description**

Installation of a fully autonomous Conda/Mamba/pip environment, and deployment of CUDA + PyTorch.

Mamba is prefered as Conda for perfomance, and using the miniforge3 distribution is the easyest way to install Mamba.

"Autonomous" means that all the code, librairies, dependencies, caches, ... will be deployed inside a directory :

- nothing is shared with the system, the environment is fully independant.
- no specific administratif rights is need.

Optional: ipython to run notebooks.

---
## **Installing Mamba**

- Define the root folder to deploy the environment : here, creating a folder "pytorch-code":
```bash
cd $HOME
mkdir pytorch-code;
cd pytorch-code
# pytorch-code environment -> PT_ENV
PT_ENV_PATH=$PWD
export PT_ENV_PATH
echo $PT_ENV_PATH
# /home/local-user/pytorch-code
```

Download and install miniforge3
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-Linux-x86_64.sh -b -p "${PT_ENV_PATH}/miniforge3"
```
Activate conda and mamba
```bash
source "${PT_ENV_PATH}/miniforge3/etc/profile.d/conda.sh"
source "${PT_ENV_PATH}/miniforge3/etc/profile.d/mamba.sh"
```

- Configure the default Conda environment
```bash
conda config --system --set channel_priority strict
conda config --system --remove-key channels
conda config --system --add channels  defaults
conda config --system --prepend channels conda-forge
conda config --system --remove channels  defaults
conda config --system --append channels  nodefaults
```

---
## **Creating an Environment**

- Create the environment: Be sure to choose the correct Python version
```bash
mamba create --name "pytorch-code" python=3.12 -y
```
- Install pip
```bash
mamba install pip
```
- Display Python versions before activating the environment
```bash
python3 --version
# Python 3.10.12

which python3
# /usr/bin/python3

python3 -m pip --version
pip 25.0.1 from /home/local-user/.local/lib/python3.10/site-packages/pip (python 3.10)
```
- Activate the environment
```bash
mamba activate pytorch-code
```
- Verify that the Python and pip versions from the environment are being used
```bash
python3 --version
# Python 3.12.9

which python3
# /home/local-user/pytorch-code/miniforge3/envs/pytorch-code/bin/python3

```
- Configure Conda for this environment
```bash
conda config --env --set channel_priority flexible
conda config --env --remove-key channels
conda config --env --add channels  defaults
conda config --env --prepend channels conda-forge
conda config --env --remove channels  defaults
conda config --env --append channels  nodefaults
```

---
## **Configuring pip**

### Cache Management

- Configuration
```bash
mkdir -p ${PT_ENV_PATH}/miniforge3/pip/cache

pip config --site set global.cache-dir  ${PT_ENV_PATH}/miniforge3/pip/cache
# Writing to /home/local-user/my-dev/miniforge3/envs/4Dvarnet-test/pip.conf
```
- Verification
```bash
pip cache dir
# /home/local-user/my-dev/miniforge3/pip/cache
```
```ini
[install]
no-user = true
```

---
## Verifying the Configuration

The packages and versions may vary depending on the Python version. However, the lists of packages installed via mamba and pip should be close to the lists below:

- mamba
```bash
mamba list
# # packages in environment at /home/local-user/pytorch-code/miniforge3/envs/pytorch-code:
# #
# # Name                    Version                   Build  Channel
# _libgcc_mutex             0.1                 conda_forge    conda-forge
# pip                       25.0.1             pyh8b19718_0    conda-forge
# python                    3.12.9          h9e4cc4f_1_cpython    conda-forge
...
```
- pip
```bash
pip freeze
# setuptools==75.8.2
# wheel==0.45.1
```
---
## **Installing CUDA + PyTorch**

`mamba` allows manual package installation, but the best practice is to use an environment file: `environment.yaml`

- If no `environment.yaml` file available, create it, ie :
```yaml
channels:
  - conda-forge
  - pytorch
  - nvidia
  - nodefaults
dependencies:
  - pip
  - pytorch::pytorch
  - pytorch::pytorch-cuda
  - pyinterp
  - tqdm
```
- and update the environment (takes time)
```bash
mamba env update -f environment.yaml
```

Messages like the one below are related to an Internet connection issue, just rerun the command.
```bash
CondaSSLError: Encountered an SSL error. Most likely a certificate verification issue.
...
CondaHTTPError: HTTP 000 CONNECTION FAILED for url <https://conda.anaconda.org/conda-forge/linux-64/libcufft-11.2.1.3-he02047a_2.conda>
...
An HTTP error occurred when trying to retrieve this URL. HTTP errors are often intermittent, and a simple retry will get you on your way.
```

---
## **Jupyterlab for Notebooks**

```bash
pip install jupyterlab
```


---
## **Known Issues**

---
### ***`AttributeError('`np.Inf` was removed in the NumPy 2.0 release. Use `np.inf` instead.'`***

- Error message : 

``` bash
hydra.errors.InstantiationException: Error in call to target 'pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint':
AttributeError('`np.Inf` was removed in the NumPy 2.0 release. Use `np.inf` instead.')
full_key: entrypoints[1].trainer.callbacks2
```

- Solution : Numpy must be downgraded to version minor than 2.0

``` bash
conda install 'numpy<2.0'
```

---
### ***`ImportError: /opt/conda/lib/python3.9/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`***

- Cause : 

``` bash
The reason is that PyTorch was built against an old version of MKL distribution which contains this symbol. However, this symbol got removed in MKL 2024.1.
The PyTorch binary released via conda channel was linked to MKL dynamically, so you got this error.
The PyTorch binary released via pip (pip install) was linked to MKL statically. You can switch to the pip install one to get rid of this error with MKL 2024.1.
...
```

- Solution : mkl must be downgraded to version `2024.0`
``` bash
conda install mkl==2024.0
```

Source :

- [https://github.com/pytorch/pytorch/issues/123097](https://github.com/pytorch/pytorch/issues/123097)


---
### ***`ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm)`***

- Full message :
``` bash
python main.py xp=base
# ...
# Testing: 0it [00:00, ?it/s]ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
# Error executing job with overrides: ['xp=base']
# Error in call to target 'src.test.base_test':
# RuntimeError('DataLoader worker (pid(s) 2387) exited unexpectedly')
# full_key: entrypoints1
```

- Cause
When working with worker, shm memory is needed, to allow workers to share datas. 

Solution :

1. Set `num_workers` to `0`, in the experience file. So worker  will not be used. It can be slow.

``` bash
{batch_size: 16, num_workers: 0}
```
2. Update SHM memory in the system if it is possible. About 512k must be enough.

To show SHM, use the command :
``` bash
df -h /dev/shm
```

---
### ***OutOfMemoryError('CUDA out of memory ...***

- Full message
``` bash
OutOfMemoryError('CUDA out of memory. Tried to allocate 1.65 GiB (GPU 0; 47.74 GiB total capacity; 41.57 GiB already allocated; 665.69 MiB free; 43.08 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF')
full_key: entrypoints1
```

- Solution : reduce the value of `batch_size` in the experience file.
``` bash
{batch_size: 4, num_workers: 1}
```

To show the GPU memory, use the command `nvidia-smi`command.



