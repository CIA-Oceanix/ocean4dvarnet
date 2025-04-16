# Code Code Quality

To run the Code Quality and Code Security suite after installing Ocean4DVarNet : 

- Install dev dependencies (pylint, black, ...) in the Ocean4DVarNet directory :
``` bash
pip install .[dev]
```
- And then run :
    - The code quality tests
``` bash
make lint
```
    - The code security tests:
```bash
make security
```

The code quality and code security tests are configured in the `pyproject.py` file, and are managed as targets in the `Makefile` file.

The code quality tools are :

- For code quality checks: [`pylint`](https://pylint.readthedocs.io/en/latest/) and [`ruff`](https://docs.astral.sh/ruff/)
- For code formatting : [`isort`](https://pycqa.github.io/isort/), [`black`](https://black.readthedocs.io/en/stable/) and [`ruff`](https://docs.astral.sh/ruff/)
- For style guide enforcement : [`flake8`](https://flake8.pycqa.org/en/latest/)
- for type checking : [`mypy`](https://mypy.readthedocs.io/en/stable/index.html)
- for docstring style checks : [`pydocstyle`](https://www.pydocstyle.org/en/stable/)

The code security are : [`bandit`](https://bandit.readthedocs.io/en/latest/) and [`safety`](https://docs.safetycli.com/safety-docs)

