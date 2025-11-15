# NNUE Weights Directory

This directory contains the trained NNUE model weights.

## Files:
- `nnue_model.pt` - Trained NNUE model state dict (PyTorch format)

## Usage:
The local NNUE implementation will automatically load weights from this directory.
If no weights file exists, it will use an untrained model with basic evaluation.
