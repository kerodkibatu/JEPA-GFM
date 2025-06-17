"""
Data generation script for JEPA-GFM

This script generates synthetic geospatial data with different modalities and resolutions
for training and testing the JEPA Geospatial Foundation Model.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import os
from tqdm import tqdm


class GeospatialDataGenerator:
    """Generator for synthetic geospatial data with multiple modalities"""
    
    def __init__(self, base_size: int = 256):
        self.base_size = base_size
        
    def generate_elevation_data(self, size: int) -> np.ndarray:
        """Generate synthetic elevation data using Perlin-like noise"""
        
        # Create coordinate grids
        x = np.linspace(0, 4 * np.pi, size)
        y = np.linspace(0, 4 * np.pi, size)
        X, Y = np.meshgrid(x, y)
        
        # Generate elevation using multiple octaves of sine waves
        elevation = np.zeros((size, size))
        
        # Multiple frequency components for realistic terrain
        frequencies = [1, 2, 4, 8]
        amplitudes = [1.0, 0.5, 0.25, 0.125]
        
        for freq, amp in zip(frequencies, amplitudes):
            elevation += amp * (
                np.sin(freq * X) * np.cos(freq * Y) +
                0.5 * np.sin(2 * freq * X + np.pi/4) * np.cos(2 * freq * Y + np.pi/4)
            )
        
        # Normalize to [0, 1]
        elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())
        
        return elevation
    
    def generate_vegetation_index(self, size: int, elevation: np.ndarray = None) -> np.ndarray:
        """Generate synthetic vegetation index (NDVI-like)"""
        
        if elevation is None:
            elevation = self.generate_elevation_data(size)
        
        # Vegetation typically decreases with elevation and has spatial clustering
        base_vegetation = 1.0 - 0.3 * elevation  # Decrease with elevation
        
        # Add spatial clustering using random fields
        np.random.seed(42)  # For reproducibility
        noise = np.random.randn(size, size)
        
        # Apply Gaussian filter for spatial correlation
        from scipy.ndimage import gaussian_filter
        vegetation_noise = gaussian_filter(noise, sigma=size/20)
        vegetation_noise = (vegetation_noise - vegetation_noise.mean()) / vegetation_noise.std()
        
        vegetation = base_vegetation + 0.1 * vegetation_noise
        vegetation = np.clip(vegetation, 0, 1)
        
        return vegetation
    
    def generate_rgb_image(self, size: int, elevation: np.ndarray = None, 
                          vegetation: np.ndarray = None) -> np.ndarray:
        """Generate synthetic RGB image based on elevation and vegetation"""
        
        if elevation is None:
            elevation = self.generate_elevation_data(size)
        if vegetation is None:
            vegetation = self.generate_vegetation_index(size, elevation)
        
        # Initialize RGB channels
        rgb = np.zeros((3, size, size))
        
        # Red channel: influenced by elevation (rock/soil at high elevation)
        rgb[0] = 0.4 + 0.4 * elevation + 0.2 * (1 - vegetation)
        
        # Green channel: influenced by vegetation
        rgb[1] = 0.3 + 0.5 * vegetation + 0.1 * (1 - elevation)
        
        # Blue channel: water bodies and atmospheric effects
        water_mask = (elevation < 0.2) & (vegetation < 0.3)
        rgb[2] = 0.2 + 0.3 * water_mask + 0.1 * elevation
        
        # Add some noise for realism
        np.random.seed(123)
        noise = np.random.randn(3, size, size) * 0.05
        rgb += noise
        
        # Clip to valid range
        rgb = np.clip(rgb, 0, 1)
        
        return rgb
    
    def generate_multispectral_bands(self, size: int, rgb: np.ndarray = None, 
                                   vegetation: np.ndarray = None) -> np.ndarray:
        """Generate additional multispectral bands (NIR, SWIR, etc.)"""
        
        if rgb is None:
            elevation = self.generate_elevation_data(size)
            vegetation = self.generate_vegetation_index(size, elevation)
            rgb = self.generate_rgb_image(size, elevation, vegetation)
        
        if vegetation is None:
            elevation = self.generate_elevation_data(size)
            vegetation = self.generate_vegetation_index(size, elevation)
        
        # Generate additional bands
        bands = np.zeros((3, size, size))  # NIR, SWIR1, SWIR2
        
        # NIR: High vegetation reflectance
        bands[0] = 0.3 + 0.6 * vegetation + 0.1 * rgb[1]
        
        # SWIR1: Sensitive to moisture content
        moisture = 1.0 - 0.5 * vegetation - 0.3 * rgb[2]  # Inverse of water
        bands[1] = 0.4 + 0.4 * moisture
        
        # SWIR2: Similar to SWIR1 but different wavelength response
        bands[2] = 0.3 + 0.5 * moisture + 0.1 * (1 - vegetation)
        
        # Add noise
        np.random.seed(456)
        noise = np.random.randn(3, size, size) * 0.03
        bands += noise
        
        bands = np.clip(bands, 0, 1)
        
        return bands
    
    def generate_sample(self, modalities: List[str] = ['rgb', 'elevation', 'vegetation', 'multispectral'],
                       size: int = None) -> Dict[str, np.ndarray]:
        """Generate a complete geospatial sample with multiple modalities"""
        
        if size is None:
            size = self.base_size
        
        # Generate base data
        elevation = self.generate_elevation_data(size)
        vegetation = self.generate_vegetation_index(size, elevation)
        rgb = self.generate_rgb_image(size, elevation, vegetation)
        multispectral = self.generate_multispectral_bands(size, rgb, vegetation)
        
        # Create output dictionary
        sample = {}
        
        if 'elevation' in modalities:
            # Convert to 3-channel for consistency
            sample['elevation'] = np.stack([elevation, elevation, elevation], axis=0)
            
        if 'vegetation' in modalities:
            # Convert to 3-channel for consistency
            sample['vegetation'] = np.stack([vegetation, vegetation, vegetation], axis=0)
            
        if 'rgb' in modalities:
            sample['rgb'] = rgb
            
        if 'multispectral' in modalities:
            sample['multispectral'] = multispectral
        
        return sample
    
    def generate_dataset(self, num_samples: int, output_dir: str = 'synthetic_data',
                        modalities: List[str] = ['rgb', 'elevation', 'vegetation', 'multispectral'],
                        size: int = None) -> None:
        """Generate a dataset of synthetic geospatial samples"""
        
        if size is None:
            size = self.base_size
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating {num_samples} synthetic geospatial samples...")
        
        for i in tqdm(range(num_samples)):
            # Generate sample with slight variations
            np.random.seed(i)  # Different seed for each sample
            
            sample = self.generate_sample(modalities, size)
            
            # Save each modality
            for modality, data in sample.items():
                modality_dir = os.path.join(output_dir, modality)
                os.makedirs(modality_dir, exist_ok=True)
                
                # Convert to tensor and save
                tensor_data = torch.from_numpy(data.astype(np.float32))
                torch.save(tensor_data, os.path.join(modality_dir, f'sample_{i:05d}.pt'))
        
        print(f"Dataset saved to {output_dir}")
    
    def visualize_sample(self, sample: Dict[str, np.ndarray], save_path: str = None):
        """Visualize a geospatial sample with all modalities"""
        
        num_modalities = len(sample)
        fig, axes = plt.subplots(2, (num_modalities + 1) // 2, figsize=(15, 10))
        
        if num_modalities == 1:
            axes = [axes]
        elif num_modalities <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for idx, (modality, data) in enumerate(sample.items()):
            if idx >= len(axes):
                break
                
            if modality in ['rgb', 'multispectral']:
                # For multi-channel data, display as RGB
                img = np.transpose(data, (1, 2, 0))
                if modality == 'multispectral':
                    # Use first 3 bands as RGB
                    img = img[:, :, :3]
            else:
                # For single-channel data (elevation, vegetation)
                img = data[0]  # Take first channel
                
            im = axes[idx].imshow(img, cmap='viridis' if modality in ['elevation', 'vegetation'] else None)
            axes[idx].set_title(f'{modality.capitalize()}')
            axes[idx].axis('off')
            
            # Add colorbar for single-channel data
            if modality in ['elevation', 'vegetation']:
                plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for idx in range(len(sample), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main function to generate synthetic geospatial data"""
    
    # Initialize generator
    generator = GeospatialDataGenerator(base_size=224)
    
    # Generate a sample for visualization
    print("Generating sample for visualization...")
    sample = generator.generate_sample(
        modalities=['rgb', 'elevation', 'vegetation', 'multispectral'],
        size=224
    )
    
    # Visualize the sample
    generator.visualize_sample(sample, 'sample_visualization.png')
    
    # Generate a small dataset
    print("\nGenerating synthetic dataset...")
    generator.generate_dataset(
        num_samples=100,
        output_dir='synthetic_geospatial_data',
        modalities=['rgb', 'elevation', 'vegetation', 'multispectral'],
        size=224
    )
    
    # Generate samples with different resolutions
    print("\nGenerating multi-resolution samples...")
    resolutions = [128, 224, 512]
    
    for res in resolutions:
        print(f"Generating sample at {res}x{res} resolution...")
        sample = generator.generate_sample(
            modalities=['rgb', 'elevation'],
            size=res
        )
        
        # Save visualization
        generator.visualize_sample(
            sample, 
            f'sample_{res}x{res}.png'
        )
    
    print("Data generation completed!")


if __name__ == '__main__':
    main()

