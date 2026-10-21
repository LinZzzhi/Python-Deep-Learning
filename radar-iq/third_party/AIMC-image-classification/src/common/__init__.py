"""
AIMC-Spec Common Utilities
==========================

This package contains shared infrastructure used throughout the AIMC-Spec
training and dataset pipeline. These modules provide reusable utilities
for dataset loading, training loops, evaluation, spectrogram generation,
and experiment management.

All model implementations and experiment scripts depend on these modules.

----------------------------------------------------------------------
Purpose
----------------------------------------------------------------------

The goal of this package is to centralise common functionality so that:

• Model training scripts remain simple and readable
• Code duplication across algorithms is avoided
• Dataset handling and evaluation remain consistent across experiments

Most experiment files should only configure models and hyperparameters,
while the core training pipeline is handled here.

----------------------------------------------------------------------
Typical Modules
----------------------------------------------------------------------

config.py
    Defines configuration dataclasses used for dataset and training
    parameters (DatasetConfig, TrainConfig).

run.py
    High-level experiment runner that orchestrates training,
    validation, checkpointing, and evaluation.

training.py
    Core training loop logic.

evaluation.py
    Validation and evaluation routines.

metrics.py
    Accuracy and performance metric utilities.

confusion_matrix.py
    Confusion matrix generation and visualisation.

memmap.py
    Utilities for reading memory-mapped datasets.

MultiHotStreamLoader.py
    Dataset loader used for generating training samples from
    AIMC-Spec emitter streams.

spectrogram.py
    Functions for generating spectrogram representations from IQ data.

checkpoint.py
    Model checkpoint saving and loading utilities.

----------------------------------------------------------------------
Important Notes
----------------------------------------------------------------------

• These modules are intended to be stable infrastructure.
• Most users should not need to modify files in this directory.
• Model-specific logic should live in the algorithm directories
  (e.g., STFT_CNN, LDC_Unet, ViT, etc.).

----------------------------------------------------------------------
Project Context
----------------------------------------------------------------------

This package is part of the AIMC-Spec framework for radar intra-pulse
modulation classification experiments.
"""