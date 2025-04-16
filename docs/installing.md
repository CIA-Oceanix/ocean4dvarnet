# Installing

Python Version: we require at least Python 3.9.

---
# Environment Installation

We currently do not provide a conda build package of Ocean4DVarNet but only a pypi package.

However, Ocean4DVarNet is based on CUDA/Pytorch. It can be complicated to install this environment from the pypi package.

So the suggested installation is through `mamba/conda` for the CUDA/Pytorch environment, and then `pip` form other dependencies.

For Linux the process to make and use a `mamba` environment is as follows :
``` bash
mamba create --name "ocean-code" python=3.12 -y
mamba activate ocean-code
```
Then the requirements must be installed in the environment :
``` bash
mamba env update -f environment.yaml
```

More informations about how to deploy a full environment can be found in the [Development environment set up](./contributing/dev-env-set-up.md) section.


---
## Instructions

To install the package and his dependencies with `pip`, you can use the following command:
``` bash
pip install ocean4dvarnet
```
or
``` bash
python -m pip install ocean4dvarnet
```


You can also get a zip or tgz archive of the package from github repository https://github.com/CIA-Oceanix/ocean4dvarnet/releases
``` bash
pip install ocean4dvarnet-contrib-x.x.x.tar.gz
```

And you can use the last development version directly from github repository
``` bash
pip install git+https://github.com/CIA-Oceanix/ocean4dvarnet.git
```

