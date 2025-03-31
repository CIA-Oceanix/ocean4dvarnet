"""
This module provides training utilities for 4D-VarNet models using PyTorch Lightning.

Functions:
    base_training: Perform basic training and testing of a model with a single datamodule.
    multi_dm_training: Perform training and testing with support for multiple datamodules.
"""

import torch
torch.set_float32_matmul_precision('high')


def base_training(trainer, dm, lit_mod, ckpt=None):
    """
    Perform basic training and testing of a model with a single datamodule.

    Args:
        trainer (pl.Trainer): The PyTorch Lightning trainer instance.
        dm (pl.LightningDataModule): The datamodule for training and testing.
        lit_mod (pl.LightningModule): The Lightning module to train.
        ckpt (str, optional): Path to a checkpoint to resume training from.

    Returns:
        None
    """
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)
    trainer.test(lit_mod, datamodule=dm, ckpt_path='best')


def multi_dm_training(
    trainer, dm, lit_mod, test_dm=None, test_fn=None, ckpt=None
):
    """
    Perform training and testing with support for multiple datamodules.

    This function trains the model using the provided datamodule and optionally tests it
    on a separate test datamodule. It also supports custom test functions for evaluation.

    Args:
        trainer (pl.Trainer): The PyTorch Lightning trainer instance.
        dm (pl.LightningDataModule): The datamodule for training.
        lit_mod (pl.LightningModule): The Lightning module to train.
        test_dm (pl.LightningDataModule, optional): The datamodule for testing. Defaults to `dm`.
        test_fn (callable, optional): A custom function to evaluate the model after testing.
        ckpt (str, optional): Path to a checkpoint to resume training from.

    Returns:
        None
    """
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)

    if test_fn is not None:
        if test_dm is None:
            test_dm = dm
        if hasattr(lit_mod, 'set_norm_stats') and callable(getattr(lit_mod, 'set_norm_stats')):
            lit_mod.set_norm_stats(test_dm.norm_stats())
        else:
            raise AttributeError("The 'lit_mod' object does not have a 'set_norm_stats' method.")

        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        trainer.callbacks = []
        trainer.test(lit_mod, datamodule=test_dm, ckpt_path=best_ckpt_path)

        print("\nBest ckpt score:")
        print(test_fn(lit_mod).to_markdown())
        print("\n###############")
