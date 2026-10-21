"""
AIMC-Spec Experiment Entry Point
================================

This file acts as the central execution entrypoint for the AIMC-Spec
training and dataset preparation pipeline.

It provides a single location where dataset generation, dataset export,
and model training experiments can be launched.

The file simply imports the `main()` functions from the relevant modules
and executes the one that is currently enabled.

----------------------------------------------------------------------
How to Run
----------------------------------------------------------------------

Run the project from the repository root using:

    python -m src.main

This ensures that Python resolves the package imports correctly.

----------------------------------------------------------------------
Typical Workflow
----------------------------------------------------------------------

The full AIMC-Spec pipeline is typically executed in stages:

1. Build the training subset (optional)
2. Export the dataset to spectrogram memmaps
3. Train a model

Only one stage is usually run at a time by enabling the corresponding
function call below.

----------------------------------------------------------------------
Pipeline Stages
----------------------------------------------------------------------

Build Training Subset
    Creates a reduced training dataset from the full emitter dataset.

    build_train_subset()

Export Memmap Spectrogram Dataset
    Converts the HDF5 IQ dataset into memmap power spectrogram files.

    export_main()

Train Model
    Runs training for a specific architecture.

    stft_main()    → STFT-CNN
    ldc_main()     → LDC-UNet
    lpi_main()     → LPI-Net
    cdae_main()    → CDAE-DCNN
    vit_main()     → Vision Transformer

Each training module defines its own dataset configuration,
model architecture, and training hyperparameters.

----------------------------------------------------------------------
How to Switch Experiments
----------------------------------------------------------------------

Only one experiment should normally be active at a time.

To run a specific model:

1. Uncomment the corresponding function call.
2. Comment out the others.
3. Run:

    python -m src.main

Example:

    stft_main()

----------------------------------------------------------------------
Outputs
----------------------------------------------------------------------

Training runs produce outputs in:

    out/<model_name>/
        training logs
        evaluation results
        confusion matrices

    checkpoints/<model_name>/
        saved model checkpoints

----------------------------------------------------------------------
Notes
----------------------------------------------------------------------

• This file intentionally contains minimal logic.
• All experiment configuration is defined in the individual modules.
• Keeping a single entrypoint simplifies experiment management and
  reproducibility.
"""

# ==========================================================
# Dataset Preparation
# ==========================================================
from src.build_iq_train_subset import main as build_train_subset
from src.export_spectrogram_memmap import main as export_main


# ==========================================================
# Model Training
# ==========================================================
from src.STFT_CNN.STFT_CNN_MemMap import main as stft_main
from src.LPI_Net.LPI_Net_MemMap import main as lpi_main
from src.LDC_Unet.LDC_Unet_MemMap import main as ldc_main
from src.CDAE_DCNN.CDAE_DCNN_MemMap import main as cdae_main
from src.ViT.Vit_MemMap import main as vit_main

if __name__ == "__main__":

    build_train_subset()

    export_main()

    stft_main() 

    # ldc_main()

    # lpi_main()

    # cdae_main()

    # vit_main()


