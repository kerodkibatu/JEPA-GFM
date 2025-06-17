"""
Training script for JEPA-GFM (JEPA Geospatial Foundation Model)

This script demonstrates how to train the JEPA model on synthetic geospatial data
with different modalities and resolutions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from typing import List, Dict, Tuple

from jepa import JEPA_GFM, create_jepa_gfm, jepa_loss


def print_gpu_memory_usage():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f'GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB')


def clear_gpu_cache():
    """Clear GPU cache to free up memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class SyntheticGeospatialDataset(Dataset):
    """
    Synthetic geospatial dataset for demonstration purposes
    Simulates different geospatial modalities like RGB, NIR, elevation, etc.
    """
    
    def __init__(self, num_samples: int = 1000, img_size: int = 224, num_modalities: int = 3):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_modalities = num_modalities
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        """
        Generate synthetic geospatial data with different modalities
        """
        layers = []
        
        for modality in range(self.num_modalities):
            # Generate different types of synthetic geospatial data
            if modality == 0:  # RGB-like data
                layer = torch.rand(3, self.img_size, self.img_size)
            elif modality == 1:  # Single channel (e.g., elevation)
                layer = torch.rand(1, self.img_size, self.img_size)
                layer = layer.repeat(3, 1, 1)  # Expand to 3 channels for consistency
            else:  # Multi-spectral data
                layer = torch.rand(3, self.img_size, self.img_size)
                
            # Add some spatial correlation to make it more realistic
            layer = torch.nn.functional.conv2d(
                layer.unsqueeze(0), 
                torch.ones(3, 1, 5, 5) / 25, 
                padding=2, 
                groups=3
            ).squeeze(0)
            
            layers.append(layer)
            
        return layers


def train_epoch(model: JEPA_GFM, dataloader: DataLoader, optimizer: optim.Optimizer, 
                device: torch.device, epoch: int, scaler: GradScaler = None) -> float:
    """Train the model for one epoch"""
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, layers in enumerate(pbar):
        # Move data to device with non_blocking for GPU efficiency
        layers = [layer.to(device, non_blocking=True) for layer in layers]
        
        # Forward pass with automatic mixed precision
        if scaler is not None:
            with autocast():
                outputs = model(layers, mask_ratio=0.75)
                loss = jepa_loss(outputs['predictions'], outputs['targets'])
            
            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(layers, mask_ratio=0.75)
            loss = jepa_loss(outputs['predictions'], outputs['targets'])
            
            # Standard backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Update target encoder with EMA
        model.update_target_encoder(momentum=0.996)
        
        # Track loss
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Log every 100 batches
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / num_batches


def validate_epoch(model: JEPA_GFM, dataloader: DataLoader, device: torch.device) -> float:
    """Validate the model"""
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for layers in tqdm(dataloader, desc='Validation'):
            # Move data to device with non_blocking for GPU efficiency
            layers = [layer.to(device, non_blocking=True) for layer in layers]
            
            # Forward pass
            outputs = model(layers, mask_ratio=0.75)
            
            # Compute loss
            loss = jepa_loss(outputs['predictions'], outputs['targets'])
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def visualize_predictions(model: JEPA_GFM, dataset: Dataset, device: torch.device, 
                         save_path: str = 'predictions.png'):
    """Visualize model predictions"""
    
    model.eval()
    
    # Get a sample from the dataset
    layers = dataset[0]
    layers = [layer.unsqueeze(0).to(device) for layer in layers]
    
    with torch.no_grad():
        outputs = model(layers, mask_ratio=0.5)
        
        # Get original image and reconstruction
        original = layers[0].squeeze(0).cpu()
        mask_indices = outputs['mask_indices'].squeeze(0).cpu()
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original.permute(1, 2, 0))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Masked image (just for visualization purposes)
        masked_img = original.clone()
        patch_size = model.context_encoder.patch_embed.patch_size
        grid_size = model.context_encoder.patch_embed.grid_size
        
        for idx in mask_indices:
            row = idx // grid_size
            col = idx % grid_size
            start_row = row * patch_size
            end_row = (row + 1) * patch_size
            start_col = col * patch_size
            end_col = (col + 1) * patch_size
            masked_img[:, start_row:end_row, start_col:end_col] = 0.5
            
        axes[1].imshow(masked_img.permute(1, 2, 0))
        axes[1].set_title('Masked Image')
        axes[1].axis('off')
        
        # Feature similarity visualization (simplified)
        predictions = outputs['predictions'].squeeze(0).cpu()
        targets = outputs['targets'].squeeze(0).cpu()
        
        # Compute cosine similarity
        pred_norm = torch.nn.functional.normalize(predictions, dim=-1)
        target_norm = torch.nn.functional.normalize(targets, dim=-1)
        similarity = torch.sum(pred_norm * target_norm, dim=-1)
        
        axes[2].bar(range(len(similarity)), similarity.numpy())
        axes[2].set_title('Patch Prediction Similarity')
        axes[2].set_xlabel('Masked Patch Index')
        axes[2].set_ylabel('Cosine Similarity')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f'Visualization saved to {save_path}')


def main():
    """Main training function"""
    
    # Configuration
    config = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 512,  # Smaller for faster training
        'encoder_depth': 8,
        'predictor_depth': 4,
        'num_heads': 8,
        'mlp_ratio': 4.0,
        'dropout': 0.1
    }
    
    # Training parameters
    batch_size = 32 if torch.cuda.is_available() else 8  # Larger batch size for GPU
    num_epochs = 10
    learning_rate = 1e-4
    num_samples = 2000 if torch.cuda.is_available() else 500  # More samples for GPU
    
    # Set device and configure for optimal GPU usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if torch.cuda.is_available():
        print(f'GPU Name: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
        print(f'CUDA Version: {torch.version.cuda}')
        # Enable optimizations for GPU
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Create datasets
    print('Creating datasets...')
    train_dataset = SyntheticGeospatialDataset(num_samples=num_samples, img_size=config['img_size'])
    val_dataset = SyntheticGeospatialDataset(num_samples=100, img_size=config['img_size'])
    
    # Create data loaders with GPU optimizations
    num_workers = 4 if torch.cuda.is_available() else 2
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Create model
    print('Creating JEPA-GFM model...')
    model = create_jepa_gfm(config)
    model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Initialize gradient scaler for mixed precision training (GPU only)
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print('Starting training...')
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch + 1, scaler)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, device)
        val_losses.append(val_loss)
        
        # Update scheduler
        scheduler.step()
        
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  LR: {scheduler.get_last_lr()[0]:.6f}')
        print_gpu_memory_usage()
        print('-' * 50)
        
        # Clear GPU cache periodically
        if (epoch + 1) % 5 == 0:
            clear_gpu_cache()
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'jepa_gfm_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('JEPA-GFM Training Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png')
    plt.close()
    print('Training curves saved to training_curves.png')
    
    # Save final model
    final_model_path = 'jepa_gfm_final.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved: {final_model_path}')
    
    # Visualize predictions
    print('Creating prediction visualization...')
    visualize_predictions(model, val_dataset, device)
    
    print('Training completed!')


if __name__ == '__main__':
    main()

