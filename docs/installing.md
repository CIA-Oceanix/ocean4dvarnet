# Installing

Python Version: we require at least Python 3.9.

---
## **Environment Installation**

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
## **Installing  from pypi**

To install the package and his dependencies with `pip`, you can use the following command:
``` bash
pip install ocean4dvarnet
```
or
``` bash
python -m pip install ocean4dvarnet
```

---
## **Installing from github repo**

You can use the last development version directly from github repository : [https://github.com/CIA-Oceanix/ocean4dvarnet](https://github.com/CIA-Oceanix/ocean4dvarnet)
``` bash
pip install git+https://github.com/CIA-Oceanix/ocean4dvarnet.git
```

You can also install a specific version or a pre-release

- From a release or tag, for example `release v1.0.6`, *see [https://github.com/CIA-Oceanix/ocean4dvarnet/releases](https://github.com/CIA-Oceanix/ocean4dvarnet/releases)*
``` bash
pip install git+https://github.com/CIA-Oceanix/ocean4dvarnet.git@v1.0.6
```
- From a specific commit, for example `commit 9f7652e`, *see [https://github.com/CIA-Oceanix/ocean4dvarnet/commits](https://github.com/CIA-Oceanix/ocean4dvarnet/commits)*
``` bash
pip install git+https://github.com/CIA-Oceanix/ocean4dvarnet.git@9f7652e
```
- From a specific branch, for example `branch 17-remove-pyinterp-dependencies`, *see [https://github.com/CIA-Oceanix/ocean4dvarnet/branches](https://github.com/CIA-Oceanix/ocean4dvarnet/branches)*
``` bash
pip install git+https://github.com/CIA-Oceanix/ocean4dvarnet.git@17-remove-pyinterp-dependencies
```

You can finally get a zip or tgz archive of the package from github repository 
``` bash
pip install ocean4dvarnet-x.x.x.tar.gz
```



