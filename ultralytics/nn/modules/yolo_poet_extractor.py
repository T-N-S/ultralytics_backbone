"""
YOLO v11 Feature Extractor for POET Integration.
Extracts multi-scale features from YOLO v11 backbone for 6D pose estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from ultralytics.nn.tasks import DetectionModel


class YOLOFeatureExtractor(nn.Module):
    """
    Feature extractor for YOLO v11 that captures intermediate backbone features
    at multiple scales for POET pose estimation transformer.
    """
    
    def __init__(self, yolo_model: DetectionModel, feature_layers: List[int] = None):
        """
        Initialize YOLO feature extractor.
        
        Args:
            yolo_model: YOLO v11 DetectionModel instance
            feature_layers: List of layer indices to extract features from.
                          If None, will auto-detect based on model architecture.
        """
        super().__init__()
        
        self.yolo_model = yolo_model
        self.features = {}
        self.hooks = []
        self.detected_strides: List[int] = []
        self.detected_channels: List[int] = []
        
        # Auto-detect feature layers if not provided
        if feature_layers is None:
            feature_layers = self._detect_feature_layers()
        
        self.feature_layers = feature_layers
        
        # Pre-detect channels/strides with a temporary pass
        self._pre_detect_channels_and_strides()
        
        # Register forward hooks to capture intermediate features
        self._register_hooks()
        
    def _detect_feature_layers(self) -> List[int]:
        """
        Auto-detect appropriate layers for feature extraction.
        Looks for layers that produce feature maps at strides 8, 16, 32.
        """
        feature_layers = []
        
        # Analyze model architecture to find appropriate layers
        for i, module in enumerate(self.yolo_model.model):
            # Look for layers that are typically at the end of each stage
            if hasattr(module, 'type'):
                # Common patterns for end-of-stage layers in YOLO
                if any(x in module.type for x in ['C2f', 'C3', 'SPPF', 'Conv']):
                    # Check if this might be a good feature extraction point
                    # by looking at position in network
                    if i in [6, 9, 12, 15, 18, 21]:  # Typical stage boundaries
                        feature_layers.append(i)
        
        # Ensure we have at least 3 feature levels
        if len(feature_layers) < 3:
            # Fallback to hardcoded values for common YOLO architectures
            feature_layers = [6, 9, 12]
            
        return feature_layers[:3]  # Take first 3 for P3, P4, P5
        
    def _pre_detect_channels_and_strides(self):
        """Temporarily hook selected layers and run a dummy pass to detect channels and strides."""
        temp_feats: Dict[int, torch.Tensor] = {}
        temp_hooks = []
        
        def make_hook(idx):
            def hook(module, inp, out):
                temp_feats[idx] = out
            return hook
        
        layers = self.yolo_model.model
        for idx in self.feature_layers:
            if idx < len(layers):
                temp_hooks.append(layers[idx].register_forward_hook(make_hook(idx)))
        
        try:
            dev = next(self.yolo_model.parameters()).device
        except StopIteration:
            dev = torch.device('cpu')
        dummy = torch.randn(1, 3, 640, 640, device=dev)
        
        with torch.no_grad():
            _ = self.yolo_model(dummy)
        
        channels: List[int] = []
        strides: List[int] = []
        input_size = 640
        for idx in self.feature_layers:
            if idx in temp_feats:
                t = temp_feats[idx]
                channels.append(int(t.shape[1]))
                strides.append(int(input_size // max(1, int(t.shape[2]))))
        
        self.detected_channels = channels
        self.detected_strides = strides
        
        for h in temp_hooks:
            h.remove()
        
    def _register_hooks(self):
        """Register forward hooks to capture features at specified layers."""
        def make_hook(layer_idx):
            def hook(module, input, output):
                self.features[layer_idx] = output
            return hook
        
        for layer_idx in self.feature_layers:
            if layer_idx < len(self.yolo_model.model):
                hook = self.yolo_model.model[layer_idx].register_forward_hook(make_hook(layer_idx))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Forward pass through YOLO model while capturing intermediate features.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            yolo_output: Final YOLO model output
            features: Dict mapping layer indices to feature tensors
        """
        # Clear previous features
        self.features.clear()
        
        # Forward pass through YOLO model
        yolo_output = self.yolo_model(x)
        
        # Return both YOLO output and captured features
        return yolo_output, self.features.copy()


class MultiScaleFeatureProcessor(nn.Module):
    """
    Processes multi-scale features from YOLO backbone into format required by POET.
    """
    
    def __init__(self, 
                 input_channels: List[int],
                 hidden_dim: int = 256,
                 num_levels: int = 3):
        """
        Initialize feature processor.
        
        Args:
            input_channels: List of input channel dimensions from backbone
            hidden_dim: Target output channel dimension
            num_levels: Number of feature pyramid levels
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.input_channels = input_channels
        
        # Create projection layers for each backbone feature level
        self.projections = nn.ModuleList()
        for channels in input_channels:
            self.projections.append(nn.Sequential(
                nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, hidden_dim),
            ))
        
        # Additional layers if we need more pyramid levels
        self.additional_layers = nn.ModuleList()
        for i in range(len(input_channels), num_levels):
            self.additional_layers.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(32, hidden_dim),
            ))
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize projection layer weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, backbone_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Process backbone features into multi-scale feature pyramid.
        
        Args:
            backbone_features: List of feature tensors from backbone
            
        Returns:
            List of processed feature tensors with consistent channel dimension
        """
        processed_features = []
        
        # Project backbone features
        for i, feat in enumerate(backbone_features):
            if i < len(self.projections):
                proj_feat = self.projections[i](feat)
                processed_features.append(proj_feat)
        
        # Generate additional pyramid levels if needed
        for i, additional_layer in enumerate(self.additional_layers):
            if i == 0 and processed_features:
                # Use last backbone feature as input for first additional layer
                new_feat = additional_layer(processed_features[-1])
            elif i > 0:
                # Use previous additional layer output
                new_feat = additional_layer(processed_features[-1])
            else:
                continue
            processed_features.append(new_feat)
        
        return processed_features[:self.num_levels]


def extract_yolo_features_for_poet(yolo_model: DetectionModel, 
                                 x: torch.Tensor,
                                 hidden_dim: int = 256,
                                 num_levels: int = 3) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Extract multi-scale features from YOLO v11 for POET integration.
    
    Args:
        yolo_model: YOLO v11 DetectionModel
        x: Input tensor of shape (B, 3, H, W)
        hidden_dim: Target feature dimension
        num_levels: Number of feature pyramid levels
        
    Returns:
        features: List of multi-scale feature tensors
        detections: YOLO detection output
    """
    
    # Create feature extractor
    extractor = YOLOFeatureExtractor(yolo_model)
    
    # Extract features and get YOLO output
    yolo_output, backbone_features = extractor(x)
    
    # Get feature tensors in order
    feature_list = []
    for layer_idx in sorted(extractor.feature_layers):
        if layer_idx in backbone_features:
            feature_list.append(backbone_features[layer_idx])
    
    # Determine input channels for processor
    if feature_list:
        input_channels = [feat.shape[1] for feat in feature_list]
    else:
        # Fallback channel dimensions for YOLO v11
        input_channels = [256, 512, 1024]  # Adjust based on model variant
        
    # Process features to consistent format
    processor = MultiScaleFeatureProcessor(input_channels, hidden_dim, num_levels)
    processed_features = processor(feature_list)
    
    # Clean up hooks
    extractor.remove_hooks()
    
    return processed_features, yolo_output


class YOLOPoETInterface:
    """
    High-level interface for integrating YOLO v11 with POET.
    """
    
    @staticmethod
    def create_poet_compatible_features(yolo_model_path: str,
                                      image_tensor: torch.Tensor,
                                      hidden_dim: int = 256,
                                      num_feature_levels: int = 3) -> Dict[str, Any]:
        """
        Create POET-compatible features from YOLO v11 model.
        
        Args:
            yolo_model_path: Path to YOLO v11 model weights
            image_tensor: Input image tensor (B, 3, H, W)
            hidden_dim: Feature projection dimension
            num_feature_levels: Number of pyramid levels
            
        Returns:
            Dictionary containing:
            - 'features': Multi-scale feature tensors
            - 'detections': YOLO detection output
            - 'masks': Attention masks for features
            - 'strides': Feature map strides
        """
        from ultralytics import YOLO
        
        # Load YOLO model
        yolo = YOLO(yolo_model_path)
        
        # Extract features
        features, detections = extract_yolo_features_for_poet(
            yolo.model, image_tensor, hidden_dim, num_feature_levels
        )
        
        # If available, read detected strides/channels from extractor
        # (Create a lightweight extractor to read metadata without extra forward)
        extractor = YOLOFeatureExtractor(yolo.model)
        detected_strides = extractor.detected_strides or [8, 16, 32][:len(features)]
        detected_channels = [hidden_dim] * len(features)
        extractor.remove_hooks()
        
        # Create attention masks (simplified - all valid)
        masks = []
        for feat in features:
            B, C, H, W = feat.shape
            mask = torch.zeros((B, H, W), dtype=torch.bool, device=feat.device)
            masks.append(mask)
        
        return {
            'features': features,
            'detections': detections,
            'masks': masks,
            'strides': detected_strides,
            'channels': detected_channels,
        }
