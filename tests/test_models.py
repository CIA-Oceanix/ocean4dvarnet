"""
Unit tests for the models and solvers in models.py using pytest.
"""

import pytest
import torch
import numpy as np
from ocean4dvarnet.models import (
    Lit4dVarNet,
    GradSolver,
    ConvLstmGradModel,
    BaseObsCost,
    BilinAEPriorCost,
)


@pytest.fixture
def mock_batch():
    """
    Create a mock batch for testing.
    """
    return {
        "input": torch.rand(2, 3, 32, 32),
        "tgt": torch.rand(2, 3, 32, 32),
    }


@pytest.fixture
def mock_solver():
    """
    Create a mock GradSolver for testing.
    """
    prior_cost = BaseObsCost()
    obs_cost = BaseObsCost()
    grad_mod = ConvLstmGradModel(dim_in=3, dim_hidden=16)
    return GradSolver(prior_cost, obs_cost, grad_mod, n_step=5)


def test_lit4dvarnet_initialization(mock_solver):
    """
    Test the initialization of Lit4dVarNet.
    """
    rec_weight = np.ones((32, 32), dtype=np.float32)
    opt_fn = lambda model: torch.optim.Adam(model.parameters(), lr=0.001)

    model = Lit4dVarNet(
        solver=mock_solver,
        rec_weight=rec_weight,
        opt_fn=opt_fn,
    )

    assert model.solver == mock_solver, "Solver not correctly assigned."
    assert model.opt_fn == opt_fn, "Optimizer function not correctly assigned."


# def test_lit4dvarnet_training_step(mock_solver, mock_batch):
#     """
#     Test the training_step method of Lit4dVarNet.
#     """
#     rec_weight = np.ones((32, 32), dtype=np.float32)
#     opt_fn = lambda model: torch.optim.Adam(model.parameters(), lr=0.001)

#     model = Lit4dVarNet(
#         solver=mock_solver,
#         rec_weight=rec_weight,
#         opt_fn=opt_fn,
#     )

#     loss = model.training_step(mock_batch, batch_idx=0)
#     assert loss is not None, "Training step did not return a loss."


def test_gradsolver_initialization():
    """
    Test the initialization of GradSolver.
    """
    prior_cost = BaseObsCost()
    obs_cost = BaseObsCost()
    grad_mod = ConvLstmGradModel(dim_in=3, dim_hidden=16)

    solver = GradSolver(prior_cost, obs_cost, grad_mod, n_step=5)
    assert solver.prior_cost == prior_cost, "Prior cost not correctly assigned."
    assert solver.obs_cost == obs_cost, "Observation cost not correctly assigned."
    assert solver.grad_mod == grad_mod, "Gradient modulation model not correctly assigned."


# def test_gradsolver_forward(mock_solver, mock_batch):
#     """
#     Test the forward method of GradSolver.
#     """
#     output = mock_solver(mock_batch)
#     assert output is not None, "GradSolver forward pass returned None."
#     assert isinstance(output, torch.Tensor), "GradSolver forward pass did not return a tensor."


def test_convlstmgradmodel_initialization():
    """
    Test the initialization of ConvLstmGradModel.
    """
    model = ConvLstmGradModel(dim_in=3, dim_hidden=16)
    assert model.dim_hidden == 16, "Hidden dimension not correctly assigned."


# def test_convlstmgradmodel_forward():
#     """
#     Test the forward method of ConvLstmGradModel.
#     """
#     model = ConvLstmGradModel(dim_in=3, dim_hidden=16)
#     x = torch.rand(2, 3, 32, 32)
#     output = model(x)
#     assert output.shape == x.shape, "ConvLstmGradModel forward pass returned incorrect shape."


# def test_baseobscost_forward(mock_batch):
#     """
#     Test the forward method of BaseObsCost.
#     """
#     cost = BaseObsCost(w=1.0)
#     state = torch.rand(2, 3, 32, 32)
#     loss = cost(state, mock_batch)
#     assert loss is not None, "BaseObsCost forward pass returned None."
#     assert isinstance(loss, torch.Tensor), "BaseObsCost forward pass did not return a tensor."


def test_bilinaepriorcost_initialization():
    """
    Test the initialization of BilinAEPriorCost.
    """
    model = BilinAEPriorCost(dim_in=3, dim_hidden=16)
    assert model.bilin_quad is True, "BilinAEPriorCost bilinear quadratic flag not correctly assigned."


def test_bilinaepriorcost_forward():
    """
    Test the forward method of BilinAEPriorCost.
    """
    model = BilinAEPriorCost(dim_in=3, dim_hidden=16)
    state = torch.rand(2, 3, 32, 32)
    loss = model(state)
    assert loss is not None, "BilinAEPriorCost forward pass returned None."
    assert isinstance(loss, torch.Tensor), "BilinAEPriorCost forward pass did not return a tensor."