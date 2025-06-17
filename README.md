# JEPA-GFM: Joint Embedding Predictive Architecture for Geospatial Foundation Model

This repository contains a Proof of Concept (POC) implementation of JEPA-GFM, a geospatial foundation model based on the Joint Embedding Predictive Architecture (JEPA). Unlike the original V-JEPA which processes temporal video frames, JEPA-GFM is designed to handle different geospatial data layers and modalities with varying resolutions.

## 🌍 Overview

JEPA-GFM adapts the powerful self-supervised learning approach of JEPA to the geospatial domain, enabling the model to:

- **Process Multiple Modalities**: Handle RGB imagery, elevation data, vegetation indices, multispectral bands, and other geospatial data types
- **Multi-Resolution Support**: Work with different spatial resolutions and automatically adapt input sizes
- **Self-Supervised Learning**: Learn rich geospatial representations without requiring labeled data
- **Foundation Model Capabilities**: Serve as a backbone for various downstream geospatial tasks

## 🏗️ Architecture

The JEPA-GFM architecture consists of three main components:

1. **Context Encoder**: Processes visible patches from geospatial data
2. **Target Encoder**: Encodes target patches with exponential moving average updates
3. **Predictor Network**: Predicts representations of masked patches in the feature space

### Key Features

- **Patch-based Processing**: Divides geospatial images into patches for efficient processing
- **Multi-Head Attention**: Captures spatial relationships between different regions
- **Modality Fusion**: Combines information from multiple geospatial data sources
- **Adaptive Resolution**: Handles variable input sizes through adaptive pooling

## 📁 Project Structure

```
GettingStarted/
├── jepa.py              # Core JEPA-GFM model implementation
├── train.py             # Training script with synthetic data
├── generate_data.py     # Synthetic geospatial data generation
├── test.py              # Comprehensive testing and evaluation
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🚀 Getting Started

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd GettingStarted
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start

#### 1. Generate Synthetic Data
```bash
python generate_data.py
```
This creates synthetic geospatial data with multiple modalities (RGB, elevation, vegetation, multispectral).

#### 2. Train the Model
```bash
python train.py
```
Trains the JEPA-GFM model on synthetic data with configurable parameters.

#### 3. Test the Model
```bash
python test.py
```
Runs comprehensive tests including multi-resolution handling, modality fusion, and downstream tasks.

## 🔧 Usage Examples

### Basic Model Usage

```python
from jepa import create_jepa_gfm
import torch

# Create model
model = create_jepa_gfm({
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': 768,
    'encoder_depth': 12,
    'predictor_depth': 6
})

# Prepare geospatial data (list of tensors for different modalities)
rgb_data = torch.randn(1, 3, 224, 224)
elevation_data = torch.randn(1, 3, 224, 224)
layers = [rgb_data, elevation_data]

# Training mode: mask and predict
model.train()
outputs = model(layers, mask_ratio=0.75)

# Inference mode: encode for downstream tasks
model.eval()
features = model.encode_geospatial_data(layers)
```

### Custom Data Integration

```python
from generate_data import GeospatialDataGenerator

# Generate custom synthetic data
generator = GeospatialDataGenerator(base_size=256)
sample = generator.generate_sample(
    modalities=['rgb', 'elevation', 'vegetation'],
    size=256
)

# Convert to tensors for model input
layers = [torch.from_numpy(data).unsqueeze(0) for data in sample.values()]
```

## 🧪 Model Configuration

The model supports various configuration options:

```python
config = {
    'img_size': 224,           # Input image size
    'patch_size': 16,          # Patch size for tokenization
    'embed_dim': 768,          # Embedding dimension
    'encoder_depth': 12,       # Number of encoder layers
    'predictor_depth': 6,      # Number of predictor layers
    'num_heads': 12,           # Number of attention heads
    'mlp_ratio': 4.0,          # MLP expansion ratio
    'dropout': 0.1             # Dropout rate
}
```

## 📊 Performance Characteristics

### Supported Modalities
- ✅ RGB imagery
- ✅ Elevation/DEM data
- ✅ Vegetation indices (NDVI-like)
- ✅ Multispectral bands (NIR, SWIR, etc.)
- ✅ Custom single/multi-channel data

### Resolution Support
- ✅ 128×128 to 512×512 pixels (tested)
- ✅ Automatic resizing for different input sizes
- ✅ Patch-based processing for memory efficiency

### Model Sizes
- **Small**: 512 dim, 8 layers (~25M parameters)
- **Base**: 768 dim, 12 layers (~85M parameters)
- **Large**: 1024 dim, 16 layers (~300M parameters)

## 🔬 Technical Details

### Self-Supervised Learning Objective

JEPA-GFM uses a joint embedding objective where:
1. Context patches are encoded to create representations
2. Target patches are encoded separately (with EMA updates)
3. A predictor network learns to predict target representations from context
4. Loss is computed in the representation space (not pixel space)

### Multi-Modal Fusion

The model handles multiple modalities through:
- Separate patch embedding for each modality
- Shared transformer backbone
- Cross-modal attention mechanisms
- Feature aggregation for downstream tasks

### Training Strategy

- **Masking Strategy**: Random patch masking (75% default)
- **Optimization**: AdamW with cosine annealing
- **Target Network Updates**: Exponential moving average (0.996 momentum)
- **Data Augmentation**: Built-in through synthetic data generation

## 🎯 Applications

JEPA-GFM can be applied to various geospatial tasks:

- **Land Cover Classification**: Classify different terrain types
- **Change Detection**: Identify changes over time
- **Environmental Monitoring**: Track vegetation, water bodies, etc.
- **Disaster Response**: Analyze affected areas
- **Urban Planning**: Monitor urban development
- **Agriculture**: Crop monitoring and yield prediction

## 📈 Evaluation

The test suite includes:

- **Multi-Resolution Handling**: Tests input size flexibility
- **Modality Fusion**: Evaluates multi-modal integration
- **Reconstruction Quality**: Measures patch prediction accuracy
- **Downstream Tasks**: Land cover classification example
- **Attention Visualization**: Feature map analysis

Run `python test.py` for comprehensive evaluation.

## 🚧 Limitations & Future Work

### Current Limitations
- Synthetic data only (for demonstration)
- Limited to square images
- Simple modality fusion strategy
- No temporal modeling

### Future Enhancements
- Real geospatial data integration
- Advanced fusion mechanisms
- Spatio-temporal modeling
- Hierarchical multi-scale processing
- Integration with existing GIS workflows

## 📚 References

1. **I-JEPA**: Assran, M., et al. "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture." CVPR 2023.
2. **V-JEPA**: Bardes, A., et al. "Revisiting Feature Prediction for Learning Visual Representations from Video." arXiv 2023.
3. **Vision Transformer**: Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.

## 🤝 Contributing

This is a POC implementation. Contributions welcome for:
- Real data integration
- Performance optimizations
- Additional modalities support
- Downstream task implementations
- Documentation improvements

## 📄 License

This project is provided as-is for educational and research purposes. Please ensure compliance with any applicable licenses when using real geospatial data.

---

**Note**: This is a Proof of Concept implementation designed for the UN Hackathon. The model architecture and training procedures can be adapted for specific geospatial applications and datasets. 