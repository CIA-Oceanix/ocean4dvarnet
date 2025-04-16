# Development environment set up

1. Create and activate a virtual environment with `conda/mamba`, and a python version >=3.9, and <3.12. *see [Setting up a CUDA + PyTorch environment wtih Mamba](./pytorch-mamba-env.md)*
2. Clone your fork repository *see [Setting Up Your Fork](./fork.md)*
3. Navigate to the repository you cloned and for which you want to install the dependencies.
``` bash
cd ocean4dvarnet
```
4. Install package and his depencies in "editable mode", then any change on the code will be seen by python interpreter so that you can see and test your modifications.
``` bash
pip install -e .
```
1. Install development dependencies
``` bash
pip install -e .[dev]
```
   - The `-e` option tells pip to install the package in editable mode, meaning that any change done to the source will be reported to the installed package.
   - The `.[dev]` argument indicate to install the current folder as a package alongside its optional dependancies, declared as dev in the pyproject.toml file. The quotes around [dev] just allow to escape the sequence in order to not be interpreted by the shell.



