# Documentation

---
## Where to update documentation


**Code documentation** :  for each Pull Request, documentation is automatically built and hosted on Github Pages.

**Global documentation** : please, open an [issue on github](https://github.com/CIA-Oceanix/ocean4dvarnet/issues)

---
## Docstrings format

Docstrings must follow the [Google's format](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings).

- Keep docstrings clear and concise while being informative.
- Include examples for non-obvious functionality.
- Document exceptions that might be raised.
- Update docstrings when changing function signatures.
- Use proper indentation in docstrings for readability.
- Add inline comments for complex logic or algorithms.

---
## Module Docstrings

Each module should start with a docstring explaining its purpose :
``` bash
"""
This module defines models and solvers for 4D-VarNet.

4D-VarNet is a framework for solving inverse problems in data assimilation 
using deep learning and PyTorch Lightning.

Classes:
    Lit4dVarNet: A PyTorch Lightning module for training and testing 4D-VarNet models.
    ...
"""
```

---
## Class Docstrings

Classes should have detailed docstrings following :
``` bash
class Lit4dVarNet(pl.LightningModule):
    """
    A PyTorch Lightning module for training and testing 4D-VarNet models.

    Attributes:
        solver (GradSolver): The solver used for optimization.
        rec_weight (torch.Tensor): Reconstruction weight for loss computation.
        opt_fn (callable): Function to configure the optimizer.
        test_metrics (dict): Dictionary of test metrics.
        pre_metric_fn (callable): Preprocessing function for metrics.
        norm_stats (tuple): Normalization statistics (mean, std).
        persist_rw (bool): Whether to persist reconstruction weight as a buffer.
    """
```

---
## Function Docstrings

Functions should have clear docstrings with parameters, returns, and examples:
``` bash
    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (dict): Input batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Training loss.
        """
```

---
## Private Methods Docstrings

Even private methods should have basic documentation, like function.


