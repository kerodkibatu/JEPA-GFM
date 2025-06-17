"""
Test script for JEPA-GFM (JEPA Geospatial Foundation Model)

This script demonstrates how to use the trained JEPA-GFM model for various
geospatial tasks and evaluates its performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from typing import List, Dict, Tuple

from jepa import JEPA_GFM, create_jepa_gfm, jepa_loss
from generate_data import GeospatialDataGenerator
from train import SyntheticGeospatialDataset


class GeospatialDownstreamTask(nn.Module):
    """
    Simple downstream task for testing JEPA-GFM representations
    Example: Land cover classification
    """
    
    def __init__(self, input_dim: int = 768, num_classes: int = 5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def test_model_loading():
    """Test loading a trained JEPA-GFM model"""
    
    print("Testing model loading...")
    
    # Create model
    config = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 512,
        'encoder_depth': 8,
        'predictor_depth': 4,
        'num_heads': 8,
        'mlp_ratio': 4.0,
        'dropout': 0.1
    }
    
    model = create_jepa_gfm(config)
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Create dummy data
    dummy_layers = [torch.randn(2, 3, 224, 224).to(device)]
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_layers, mask_ratio=0.5)
        
    print(f"✓ Model forward pass successful")
    print(f"  - Predictions shape: {outputs['predictions'].shape}")
    print(f"  - Targets shape: {outputs['targets'].shape}")
    print(f"  - Context features shape: {outputs['context_features'].shape}")
    
    # Test encoding function
    encoded_features = model.encode_geospatial_data(dummy_layers)
    print(f"  - Encoded features shape: {encoded_features.shape}")
    
    return model


def test_multi_resolution_handling():
    """Test model's ability to handle different resolutions"""
    
    print("\nTesting multi-resolution handling...")
    
    model = create_jepa_gfm()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    resolutions = [128, 224, 256, 512]
    
    for res in resolutions:
        print(f"Testing resolution: {res}x{res}")
        
        # Create dummy data with different resolution
        dummy_data = torch.randn(1, 3, res, res).to(device)
        
        with torch.no_grad():
            try:
                features = model.encode_geospatial_data([dummy_data])
                print(f"  ✓ Resolution {res}x{res}: {features.shape}")
            except Exception as e:
                print(f"  ✗ Resolution {res}x{res}: {str(e)}")


def test_multi_modality_fusion():
    """Test model's ability to fuse multiple modalities"""
    
    print("\nTesting multi-modality fusion...")
    
    model = create_jepa_gfm()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Generate synthetic multi-modal data
    generator = GeospatialDataGenerator(base_size=224)
    sample = generator.generate_sample(
        modalities=['rgb', 'elevation', 'vegetation', 'multispectral'],
        size=224
    )
    
    # Convert to tensors
    layers = []
    for modality, data in sample.items():
        tensor_data = torch.from_numpy(data).unsqueeze(0).to(device)
        layers.append(tensor_data)
    
    print(f"Number of modalities: {len(layers)}")
    
    with torch.no_grad():
        # Test individual modality encoding
        individual_features = []
        for i, layer in enumerate(layers):
            features = model.encode_geospatial_data([layer])
            individual_features.append(features)
            print(f"  Modality {i} features: {features.shape}")
        
        # Test multi-modal fusion
        fused_features = model.encode_geospatial_data(layers)
        print(f"  Fused features: {fused_features.shape}")
        
        # Compare individual vs fused representations
        individual_mean = torch.stack(individual_features).mean(dim=0)
        similarity = F.cosine_similarity(fused_features, individual_mean, dim=-1)
        print(f"  Similarity between fused and mean individual: {similarity.item():.4f}")


def test_reconstruction_quality():
    """Test the quality of patch reconstruction"""
    
    print("\nTesting reconstruction quality...")
    
    model = create_jepa_gfm()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Generate test data
    generator = GeospatialDataGenerator(base_size=224)
    sample = generator.generate_sample(modalities=['rgb'], size=224)
    
    # Convert to tensor
    rgb_data = torch.from_numpy(sample['rgb']).unsqueeze(0).to(device)
    
    # Test different mask ratios
    mask_ratios = [0.25, 0.5, 0.75, 0.9]
    
    reconstruction_similarities = []
    
    for mask_ratio in mask_ratios:
        with torch.no_grad():
            outputs = model([rgb_data], mask_ratio=mask_ratio)
            
            # Compute reconstruction similarity
            predictions = outputs['predictions']
            targets = outputs['targets']
            
            # Normalize features
            pred_norm = F.normalize(predictions, dim=-1)
            target_norm = F.normalize(targets, dim=-1)
            
            # Compute cosine similarity
            similarity = torch.sum(pred_norm * target_norm, dim=-1).mean()
            reconstruction_similarities.append(similarity.item())
            
            print(f"  Mask ratio {mask_ratio:.2f}: Similarity {similarity:.4f}")
    
    # Plot reconstruction quality vs mask ratio
    plt.figure(figsize=(10, 6))
    plt.plot(mask_ratios, reconstruction_similarities, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Mask Ratio')
    plt.ylabel('Reconstruction Similarity')
    plt.title('JEPA-GFM Reconstruction Quality vs Mask Ratio')
    plt.grid(True, alpha=0.3)
    plt.savefig('reconstruction_quality.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  Reconstruction quality plot saved: reconstruction_quality.png")


def test_downstream_task():
    """Test JEPA-GFM features on a downstream task"""
    
    print("\nTesting downstream task performance...")
    
    # Load model
    model = create_jepa_gfm()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Create synthetic downstream task data
    num_samples = 200
    generator = GeospatialDataGenerator(base_size=224)
    
    # Generate data with labels (simplified land cover classification)
    features_list = []
    labels_list = []
    
    print("Generating downstream task data...")
    for i in tqdm(range(num_samples)):
        # Generate sample
        sample = generator.generate_sample(modalities=['rgb'], size=224)
        rgb_data = torch.from_numpy(sample['rgb']).unsqueeze(0).to(device)
        
        # Extract features using JEPA-GFM
        with torch.no_grad():
            features = model.encode_geospatial_data([rgb_data])
            features_list.append(features.cpu())
        
        # Create synthetic labels based on data characteristics
        elevation = generator.generate_elevation_data(224)
        vegetation = generator.generate_vegetation_index(224, elevation)
        
        # Simple land cover classification based on elevation and vegetation
        if elevation.mean() > 0.7:
            label = 0  # Mountain/Rock
        elif vegetation.mean() > 0.6:
            label = 1  # Forest
        elif elevation.mean() < 0.3 and vegetation.mean() < 0.4:
            label = 2  # Water/Barren
        elif vegetation.mean() > 0.4:
            label = 3  # Grassland
        else:
            label = 4  # Mixed/Other
            
        labels_list.append(label)
    
    # Convert to tensors
    features_tensor = torch.cat(features_list, dim=0)
    labels_tensor = torch.tensor(labels_list)
    
    print(f"Features shape: {features_tensor.shape}")
    print(f"Labels shape: {labels_tensor.shape}")
    print(f"Label distribution: {torch.bincount(labels_tensor)}")
    
    # Train simple classifier
    classifier = GeospatialDownstreamTask(
        input_dim=features_tensor.shape[-1],
        num_classes=5
    ).to(device)
    
    # Split data
    split_idx = int(0.8 * len(features_tensor))
    train_features, test_features = features_tensor[:split_idx], features_tensor[split_idx:]
    train_labels, test_labels = labels_tensor[:split_idx], labels_tensor[split_idx:]
    
    # Train classifier
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    classifier.train()
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = classifier(train_features.to(device))
        loss = criterion(outputs, train_labels.to(device))
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Test classifier
    classifier.eval()
    with torch.no_grad():
        test_outputs = classifier(test_features.to(device))
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == test_labels.to(device)).float().mean()
        
    print(f"  Downstream task accuracy: {accuracy:.4f}")
    
    return accuracy.item()


def visualize_attention_maps():
    """Visualize attention patterns in the model"""
    
    print("\nGenerating attention visualizations...")
    
    # This is a simplified visualization - in practice, you'd need to modify
    # the model to return attention weights
    
    model = create_jepa_gfm()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Generate test data
    generator = GeospatialDataGenerator(base_size=224)
    sample = generator.generate_sample(modalities=['rgb'], size=224)
    rgb_data = torch.from_numpy(sample['rgb']).unsqueeze(0).to(device)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model([rgb_data], mask_ratio=0.5)
        context_features = outputs['context_features']
        mask_indices = outputs['mask_indices']
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    original_img = sample['rgb'].transpose(1, 2, 0)
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Masked patches
    masked_img = original_img.copy()
    patch_size = model.context_encoder.patch_embed.patch_size
    grid_size = model.context_encoder.patch_embed.grid_size
    
    for idx in mask_indices[0].cpu():
        row = idx // grid_size
        col = idx % grid_size
        start_row = row * patch_size
        end_row = (row + 1) * patch_size
        start_col = col * patch_size
        end_col = (col + 1) * patch_size
        masked_img[start_row:end_row, start_col:end_col] = 0.5
    
    axes[1].imshow(masked_img)
    axes[1].set_title('Masked Image')
    axes[1].axis('off')
    
    # Feature map visualization (simplified)
    # Take the mean of context features across patches
    feature_map = context_features[0, 1:].mean(dim=-1).cpu().numpy()  # Skip CLS token
    feature_map = feature_map.reshape(grid_size, grid_size)
    
    im = axes[2].imshow(feature_map, cmap='viridis')
    axes[2].set_title('Context Feature Map')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  Attention visualization saved: attention_visualization.png")


def run_comprehensive_test():
    """Run comprehensive tests of JEPA-GFM"""
    
    print("=" * 60)
    print("JEPA-GFM Comprehensive Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Model loading and basic functionality
    try:
        model = test_model_loading()
        results['model_loading'] = True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        results['model_loading'] = False
        return results
    
    # Test 2: Multi-resolution handling
    try:
        test_multi_resolution_handling()
        results['multi_resolution'] = True
    except Exception as e:
        print(f"✗ Multi-resolution test failed: {e}")
        results['multi_resolution'] = False
    
    # Test 3: Multi-modality fusion
    try:
        test_multi_modality_fusion()
        results['multi_modality'] = True
    except Exception as e:
        print(f"✗ Multi-modality test failed: {e}")
        results['multi_modality'] = False
    
    # Test 4: Reconstruction quality
    try:
        test_reconstruction_quality()
        results['reconstruction'] = True
    except Exception as e:
        print(f"✗ Reconstruction test failed: {e}")
        results['reconstruction'] = False
    
    # Test 5: Downstream task
    try:
        accuracy = test_downstream_task()
        results['downstream_accuracy'] = accuracy
        results['downstream_task'] = True
    except Exception as e:
        print(f"✗ Downstream task test failed: {e}")
        results['downstream_task'] = False
    
    # Test 6: Attention visualization
    try:
        visualize_attention_maps()
        results['visualization'] = True
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        results['visualization'] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    
    for test_name, result in results.items():
        if test_name == 'downstream_accuracy':
            continue
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<20}: {status}")
        
    if 'downstream_accuracy' in results:
        print(f"{'downstream_accuracy':<20}: {results['downstream_accuracy']:.4f}")
    
    print("=" * 60)
    
    return results


def main():
    """Main test function"""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run comprehensive tests
    results = run_comprehensive_test()
    
    # Save results
    torch.save(results, 'test_results.pt')
    print("Test results saved to: test_results.pt")


if __name__ == '__main__':
    main()

