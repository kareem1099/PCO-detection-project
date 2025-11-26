PCO (Polycystic Ovary) Image Classification
Overview

This project implements a PCO image classification pipeline using two deep learning architectures:

TNT (Transformer in Transformer) – a vision transformer model fine-tuned for binary classification.

Simple MLP (Multi-Layer Perceptron) – a fully connected network trained on raw image pixels.

The pipeline handles:

Data augmentation

Train/validation/test splitting

Model training with early stopping and learning rate scheduling

Evaluation on a held-out test set

Project Structure
D:/pco/
│
├─ infected/                # Original infected PCO images
├─ not_infected/            # Original normal images
├─ augmented_data3/         # Augmented dataset (after transformations)
├─ final_split3/            # Train/val/test splits
├─ best_tnt_model.pth       # Best TNT model checkpoint
├─ best_mlp_model.pth       # Best MLP model checkpoint
└─ train_pipeline.py        # Main training script

Requirements

Install the required Python packages:

pip install torch torchvision timm numpy pillow matplotlib


Optional (for advanced augmentation or dataset handling):

pip install albumentations

Data Preparation

Place your dataset images into two folders:

D:/pco/infected
D:/pco/not_infected


The script automatically performs augmentation:

Infected: Random horizontal flips, rotations, color jitter, resized to 224×224

Not_Infected: Random horizontal flips, resized to 224×224

The dataset is then split into train/validation/test sets:

70% train

10% validation

20% test

Dataloaders

Images are converted to tensors and normalized with ImageNet statistics:

transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


Batch size is set to 32.

Models
1. TNT Model

Pretrained tnt_s_patch16_224 from the timm library

Backbone frozen except classification head

Dropout added to head for regularization

Optimizer: Adam

Learning rate scheduler: ReduceLROnPlateau

2. MLP Model

Fully connected network on flattened image input

Layers:

Linear(3×224×224 → 512) → ReLU → Dropout

Linear(512 → 128) → ReLU → Dropout

Linear(128 → 2)

Optimizer: Adam

Early stopping and LR scheduling similar to TNT

Training
train_tnt_model()
train_model(model_mlp, "mlp_model")


Maximum epochs: 50

Early stopping patience: 5 epochs

Model checkpoints saved at best validation accuracy:

TNT: best_tnt_model.pth

MLP: best_mlp_model.pth

Evaluation

Evaluate both models on the test set:

evaluate_model(model_tnt, "tnt_model")
evaluate_model(model_mlp, "mlp_model")


Outputs accuracy for each model.

Notes

Device: Automatically uses GPU if available, else CPU.

Augmentation: Adjustable via transforms.Compose.

Custom datasets: Update original_infected_dir and original_not_infected_dir.

Batch size and learning rates can be tuned according to dataset size.

License

MIT License
