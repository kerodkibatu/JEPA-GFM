"""
JEPA-GFM: Joint Embedding Predictive Architecture for Geospatial Foundation Model

This implementation adapts the V-JEPA architecture for geospatial data,
where instead of temporal frames, we use different layers/datasets with
varying resolutions and modalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from typing import Dict, List, Optional, Tuple


class PatchEmbedding(nn.Module):
    """Patch embedding for geospatial data with variable resolution support"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, 
                 embed_dim: int = 768, norm_layer: Optional[nn.Module] = None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Handle variable input sizes by adaptive pooling if needed
        if H != self.img_size or W != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = rearrange(x, 'b c h w -> b (h w) c')  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with layer normalization and feed-forward network"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class GeospatialEncoder(nn.Module):
    """Encoder for processing different geospatial modalities"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize parameters
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add cls token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
            
        x = self.norm(x)
        return x


class GeospatialPredictor(nn.Module):
    """Predictor network for JEPA-GFM"""
    
    def __init__(self, embed_dim: int = 768, predictor_embed_dim: int = 384, depth: int = 6,
                 num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(predictor_embed_dim)
        self.head = nn.Linear(predictor_embed_dim, embed_dim)
        
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
    def forward(self, x: torch.Tensor, mask_indices: torch.Tensor) -> torch.Tensor:
        # x: encoded context patches
        # mask_indices: indices of patches to predict
        
        B, N, C = x.shape
        x = self.predictor_embed(x)
        
        # Create mask tokens for prediction targets
        mask_tokens = repeat(self.mask_token, '1 1 d -> b n d', 
                           b=B, n=mask_indices.shape[1])
        
        # Insert mask tokens at specified positions
        x_pred = torch.zeros(B, N, self.mask_token.shape[-1], device=x.device)
        x_pred.scatter_(1, mask_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]), mask_tokens)
        
        # Keep context patches
        context_mask = torch.ones(B, N, device=x.device)
        context_mask.scatter_(1, mask_indices, 0)
        x_pred = x_pred + x * context_mask.unsqueeze(-1)
        
        # Apply predictor blocks
        for block in self.blocks:
            x_pred = block(x_pred)
            
        x_pred = self.norm(x_pred)
        x_pred = self.head(x_pred)
        
        return x_pred


class JEPA_GFM(nn.Module):
    """
    JEPA Geospatial Foundation Model
    
    Adapted from V-JEPA for geospatial data where each "frame" represents
    different layers/datasets with varying resolutions and modalities.
    """
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, embed_dim: int = 768,
                 encoder_depth: int = 12, predictor_depth: int = 6, num_heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        
        # Context encoder - encodes visible patches
        self.context_encoder = GeospatialEncoder(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=encoder_depth, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )
        
        # Target encoder - encodes target patches (no gradients)
        self.target_encoder = GeospatialEncoder(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=encoder_depth, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )
        
        # Predictor network
        self.predictor = GeospatialPredictor(
            embed_dim=embed_dim, predictor_embed_dim=embed_dim // 2,
            depth=predictor_depth, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )
        
        # Initialize target encoder with same weights as context encoder
        self._initialize_target_encoder()
        
    def _initialize_target_encoder(self):
        """Initialize target encoder with context encoder weights"""
        for param_c, param_t in zip(self.context_encoder.parameters(), 
                                   self.target_encoder.parameters()):
            param_t.data.copy_(param_c.data)
            param_t.requires_grad = False
            
    def update_target_encoder(self, momentum: float = 0.996):
        """Update target encoder with exponential moving average"""
        with torch.no_grad():
            for param_c, param_t in zip(self.context_encoder.parameters(),
                                       self.target_encoder.parameters()):
                param_t.data = momentum * param_t.data + (1 - momentum) * param_c.data
                
    def forward(self, layers: List[torch.Tensor], mask_ratio: float = 0.75) -> Dict[str, torch.Tensor]:
        """
        Forward pass for JEPA-GFM
        
        Args:
            layers: List of geospatial data tensors representing different modalities/resolutions
            mask_ratio: Ratio of patches to mask for prediction
            
        Returns:
            Dictionary containing predictions and targets
        """
        # For simplicity, we'll work with the first layer as primary and others as context
        primary_layer = layers[0]
        B, C, H, W = primary_layer.shape
        
        # Generate random mask
        num_patches = self.context_encoder.patch_embed.num_patches
        num_mask = int(num_patches * mask_ratio)
        
        # Create mask indices
        mask_indices = torch.randperm(num_patches, device=primary_layer.device)[:num_mask]
        mask_indices = repeat(mask_indices, 'n -> b n', b=B)
        
        # Encode context (visible patches)
        context_features = self.context_encoder(primary_layer)
        
        # Encode targets (full image, no gradients)
        with torch.no_grad():
            target_features = self.target_encoder(primary_layer)
            
        # Predict masked patches
        all_predictions = self.predictor(context_features, mask_indices)
        
        # Extract predictions for masked patches only
        predictions = torch.gather(all_predictions, 1, 
                                 mask_indices.unsqueeze(-1).expand(-1, -1, all_predictions.shape[-1]))
        
        # Extract targets for masked patches
        targets = torch.gather(target_features, 1, 
                             mask_indices.unsqueeze(-1).expand(-1, -1, target_features.shape[-1]))
        
        return {
            'predictions': predictions,
            'targets': targets,
            'mask_indices': mask_indices,
            'context_features': context_features
        }
    
    def encode_geospatial_data(self, layers: List[torch.Tensor]) -> torch.Tensor:
        """
        Encode geospatial data for downstream tasks
        
        Args:
            layers: List of geospatial data tensors
            
        Returns:
            Encoded features
        """
        # Process each layer and aggregate features
        layer_features = []
        
        for layer in layers:
            features = self.context_encoder(layer)
            layer_features.append(features[:, 0])  # Use CLS token
            
        # Simple aggregation - can be made more sophisticated
        if len(layer_features) > 1:
            aggregated_features = torch.stack(layer_features, dim=1).mean(dim=1)
        else:
            aggregated_features = layer_features[0]
            
        return aggregated_features


def create_jepa_gfm(config: Optional[Dict] = None) -> JEPA_GFM:
    """Factory function to create JEPA-GFM model"""
    
    default_config = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 768,
        'encoder_depth': 12,
        'predictor_depth': 6,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'dropout': 0.1
    }
    
    if config:
        default_config.update(config)
        
    return JEPA_GFM(**default_config)


# Loss function for JEPA training
def jepa_loss(predictions: torch.Tensor, targets: torch.Tensor, 
              temperature: float = 0.1) -> torch.Tensor:
    """
    Compute JEPA loss using cosine similarity
    
    Args:
        predictions: Predicted features for masked patches
        targets: Target features for masked patches
        temperature: Temperature for similarity computation
        
    Returns:
        Loss value
    """
    # Normalize features
    predictions = F.normalize(predictions, dim=-1)
    targets = F.normalize(targets, dim=-1)
    
    # Compute cosine similarity
    similarity = torch.sum(predictions * targets, dim=-1)
    
    # Apply temperature scaling
    similarity = similarity / temperature
    
    # Compute loss (negative log likelihood)
    loss = -torch.mean(similarity)
    
    return loss

