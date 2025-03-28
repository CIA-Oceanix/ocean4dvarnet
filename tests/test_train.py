"""
Unit tests for the training utilities in train.py using pytest.
"""

import pytest
from unittest.mock import MagicMock
from ocean4dvarnet.train import base_training, multi_dm_training


@pytest.fixture
def mock_trainer():
    """
    Create a mock PyTorch Lightning trainer.
    """
    trainer = MagicMock()
    trainer.logger = MagicMock()
    trainer.logger.log_dir = "/mock/logdir"
    trainer.checkpoint_callback = MagicMock()
    trainer.checkpoint_callback.best_model_path = "/mock/best_model.ckpt"
    return trainer


@pytest.fixture
def mock_datamodule():
    """
    Create a mock PyTorch Lightning datamodule.
    """
    dm = MagicMock()
    dm.norm_stats = MagicMock(return_value=(0.0, 1.0))
    return dm


@pytest.fixture
def mock_lit_module():
    """
    Create a mock PyTorch Lightning module.
    """
    lit_mod = MagicMock()
    lit_mod.set_norm_stats = MagicMock()
    return lit_mod


def test_base_training(mock_trainer, mock_datamodule, mock_lit_module):
    """
    Test the base_training function.
    """
    base_training(mock_trainer, mock_datamodule, mock_lit_module, ckpt="/mock/ckpt")
    
    # Check that trainer.fit and trainer.test were called
    mock_trainer.fit.assert_called_once_with(mock_lit_module, datamodule=mock_datamodule, ckpt_path="/mock/ckpt")
    mock_trainer.test.assert_called_once_with(mock_lit_module, datamodule=mock_datamodule, ckpt_path="best")


# def test_multi_dm_training(mock_trainer, mock_datamodule, mock_lit_module):
#     """
#     Test the multi_dm_training function.
#     """
#     test_fn = MagicMock(return_value="Test metrics")
#     multi_dm_training(mock_trainer, mock_datamodule, mock_lit_module, test_dm=mock_datamodule, test_fn=test_fn, ckpt="/mock/ckpt")
    
#     # Check that trainer.fit was called
#     mock_trainer.fit.assert_called_once_with(mock_lit_module, datamodule=mock_datamodule, ckpt_path="/mock/ckpt")
    
#     # Check that lit_mod.set_norm_stats was called
#     mock_lit_module.set_norm_stats.assert_called_once_with(mock_datamodule.norm_stats())
    
#     # Check that trainer.test was called
#     mock_trainer.test.assert_called_once_with(mock_lit_module, datamodule=mock_datamodule, ckpt_path="/mock/best_model.ckpt")
    
#     # Check that the test function was called
#     test_fn.assert_called_once_with(mock_lit_module)