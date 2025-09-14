"""
POET Adapter module for YOLO v11 integration.
Provides multi-scale feature extraction and NestedTensor conversion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any


class NestedTensor:
    """
    NestedTensor class for POET compatibility.
    Holds tensor and mask pairs for POET/DETR frameworks.
    """
    
    def __init__(self, tensors: torch.Tensor, mask: torch.Tensor):
        self.tensors = tensors
        self.mask = mask
        
    def decompose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.tensors, self.mask
        
    def to(self, device):
        return NestedTensor(self.tensors.to(device), self.mask.to(device))


class PoETAdapter(nn.Module):
    """
    Adapter to extract multi-scale features from YOLO v11 backbone 
    and convert to POET-compatible format.
    """
    
    def __init__(self, 
                 yolo_model: nn.Module,
                 hidden_dim: int = 256,
                 num_feature_levels: int = 3,
                 backbone_feature_indices: List[int] = None):
        """
        Initialize PoET adapter.
        
        Args:
            yolo_model: YOLO v11 model instance
            hidden_dim: Target channel dimension for feature projection (default: 256)
            num_feature_levels: Number of feature levels to extract (default: 3)
            backbone_feature_indices: Indices of backbone layers to extract features from
                                    If None, will automatically determine based on YOLO architecture
        """
        super().__init__()
        
        self.yolo_model = yolo_model
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        # Cache for last YOLO predictions to avoid redundant forwards
        self._last_predictions: Any = None
        
        # Feature indices (P3/P4/P5 style). Fallback to typical values.
        if backbone_feature_indices is None:
            backbone_feature_indices = self._get_backbone_feature_indices()
        self.backbone_feature_indices = backbone_feature_indices[:self.num_feature_levels]
        
        # Storage for captured features and hooks
        self._features: Dict[int, torch.Tensor] = {}
        self._hooks: List[Any] = []
        
        # Seed default strides; will be updated after detection
        self.strides = [8, 16, 32][:self.num_feature_levels]
        
        # Detect backbone channel dims and actual strides using a temporary pass
        self.backbone_channels = self._detect_channels_and_strides()
        
        # Build projection heads to map to hidden_dim
        self.input_proj = nn.ModuleList()
        
        def _gn_groups(dim: int) -> int:
            for g in (32, 16, 8, 4, 2, 1):
                if dim % g == 0:
                    return g
            return 1
        
        for ch in self.backbone_channels:
            self.input_proj.append(nn.Sequential(
                nn.Conv2d(ch, hidden_dim, kernel_size=1, bias=False),
                nn.GroupNorm(_gn_groups(hidden_dim), hidden_dim),
            ))
        # Additional levels if requested
        for _ in range(len(self.backbone_channels), self.num_feature_levels):
            self.input_proj.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(_gn_groups(hidden_dim), hidden_dim),
            ))
        
        # Init weights ONLY for adapter projection layers (do not touch YOLO weights)
        for proj in self.input_proj:
            for m in proj.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight, gain=1)
                    if getattr(m, 'bias', None) is not None:
                        nn.init.constant_(m.bias, 0)
        
        # Register persistent hooks to capture features during real forward
        self._register_persistent_hooks()
        
        # Reported to POET
        self.num_channels = [hidden_dim] * self.num_feature_levels
    
    @property
    def feature_layers(self) -> List[int]:
        """Alias to match external expectations/tests."""
        return self.backbone_feature_indices
    
    def _get_backbone_feature_indices(self) -> List[int]:
        """
        Automatically determine which layers to extract features from.
        Prefer layers whose output strides are closest to [8, 16, 32] relative to 640 input.
        Falls back to monotonically decreasing resolutions if exact strides aren't present.
        """
        layers = getattr(self.yolo_model, 'model', None)
        if layers is None:
            return [6, 9, 12]
        
        # Temporary hooks to record only output shapes (and channels) for each layer
        temp_info: Dict[int, Tuple[int, int, int]] = {}
        hooks = []
        
        def make_hook(idx: int):
            def hook(_module, _inp, out):
                # Support cases where a layer returns tuple/list
                t = None
                if isinstance(out, torch.Tensor):
                    t = out
                elif isinstance(out, (list, tuple)) and out:
                    # pick first tensor-like item
                    for o in out:
                        if isinstance(o, torch.Tensor):
                            t = o
                            break
                if t is not None and t.dim() >= 4:
                    c, h, w = int(t.shape[1]), int(t.shape[2]), int(t.shape[3])
                    temp_info[idx] = (c, h, w)
            return hook
        
        for i in range(len(layers)):
            try:
                hooks.append(layers[i].register_forward_hook(make_hook(i)))
            except Exception:
                pass
        
        # Run a dummy forward
        try:
            dev = next(self.yolo_model.parameters()).device
        except StopIteration:
            dev = torch.device('cpu')
        dummy_size = 640
        dummy = torch.randn(1, 3, dummy_size, dummy_size, device=dev)
        with torch.no_grad():
            try:
                _ = self.yolo_model(dummy)
            except Exception:
                # If model forward fails for any reason, clean up and fallback
                for h in hooks:
                    try:
                        h.remove()
                    except Exception:
                        pass
                return [6, 9, 12]
        
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass
        
        # Build candidate list with computed strides
        candidates: List[Tuple[int, int, int, int]] = []  # (idx, stride, h, c)
        for idx, (c, h, w) in temp_info.items():
            if h > 0:
                stride = max(1, int(round(dummy_size / h)))
                candidates.append((idx, stride, h, c))
        if not candidates:
            return [6, 9, 12]
        
        # Sort by stride ascending, then by index ascending
        candidates.sort(key=lambda x: (x[1], x[0]))
        
        # Select indices closest to target strides [8,16,32] while ensuring increasing stride
        target = [8, 16, 32]
        selected: List[int] = []
        used_idxs = set()
        used_strides = set()
        
        for t in target:
            # Find candidate with stride closest to target; prefer deeper layer for same stride
            best = None
            best_key = None
            for (idx, s, h, c) in candidates:
                if idx in used_idxs:
                    continue
                # enforce monotonic increasing stride
                if selected:
                    prev_stride = max(used_strides)
                    if s <= prev_stride:
                        continue
                diff = abs(s - t)
                # key: smaller diff first, then larger index (deeper), then larger stride
                key = (diff, -idx, s)
                if (best is None) or (key < best_key):
                    best = (idx, s)
                    best_key = key
            if best is not None:
                idx, s = best
                selected.append(idx)
                used_idxs.add(idx)
                used_strides.add(s)
        
        # If we couldn't pick 3, fill remaining with highest-stride unique maps
        if len(selected) < 3:
            # take remaining candidates with strictly increasing stride
            last_stride = max(used_strides) if used_strides else 0
            for (idx, s, h, c) in candidates:
                if idx in used_idxs or s <= last_stride:
                    continue
                selected.append(idx)
                used_idxs.add(idx)
                last_stride = s
                if len(selected) == 3:
                    break
        
        # Final fallback
        if len(selected) < 3:
            # choose three largest strides overall
            uniq_by_stride = {}
            for idx, s, h, c in candidates:
                uniq_by_stride[s] = idx
            selected = [uniq_by_stride[s] for s in sorted(uniq_by_stride.keys()) if s > 1][-3:]
            if not selected:
                return [6, 9, 12]
        
        # Return sorted by stride increasing (P3->P4->P5 style)
        # Map strides for selected and sort accordingly
        stride_map = {idx: s for idx, s, h, c in candidates}
        selected.sort(key=lambda i: stride_map.get(i, 1))
        return selected
    
    def _detect_channels_and_strides(self) -> List[int]:
        """Temporarily hook selected layers, run a dummy pass, capture shapes and compute channels/strides."""
        temp_feats: Dict[int, torch.Tensor] = {}
        temp_hooks = []
        
        def make_hook(idx):
            def hook(module, inp, out):
                # handle nested outputs
                t = out
                if not isinstance(out, torch.Tensor):
                    if isinstance(out, (list, tuple)) and out:
                        for o in out:
                            if isinstance(o, torch.Tensor):
                                t = o
                                break
                if isinstance(t, torch.Tensor):
                    temp_feats[idx] = t
            return hook
        
        # Register temporary hooks on actual layer indices
        model_layers = self.yolo_model.model
        for layer_idx in self.backbone_feature_indices:
            if layer_idx < len(model_layers):
                temp_hooks.append(model_layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
        
        # Build dummy input on same device as model
        try:
            dev = next(self.yolo_model.parameters()).device
        except StopIteration:
            dev = torch.device('cpu')
        dummy = torch.randn(1, 3, 640, 640, device=dev)
        
        with torch.no_grad():
            _ = self.yolo_model(dummy)
        
        channels: List[int] = []
        actual_strides: List[int] = []
        input_size = 640
        for layer_idx in self.backbone_feature_indices:
            if layer_idx in temp_feats:
                t = temp_feats[layer_idx]
                channels.append(int(t.shape[1]))
                stride = int(input_size // max(1, int(t.shape[2])))
                actual_strides.append(stride)
        
        # Update strides if detected and strictly increasing; else sort unique increasing
        if actual_strides:
            # ensure increasing order by sorting according to spatial size captured order
            # recompute order by stride
            paired = list(zip(self.backbone_feature_indices, channels, actual_strides))
            paired.sort(key=lambda x: x[2])
            self.backbone_feature_indices = [p[0] for p in paired][:self.num_feature_levels]
            channels = [p[1] for p in paired][:self.num_feature_levels]
            self.strides = [p[2] for p in paired][:self.num_feature_levels]
        
        # Cleanup temporary hooks
        for h in temp_hooks:
            h.remove()
        
        # Fallback channels if not captured
        if not channels:
            channels = [256, 512, 1024][:self.num_feature_levels]
        
        return channels
    
    def _register_persistent_hooks(self):
        """Register hooks that populate self._features per forward call."""
        def make_hook(idx):
            def hook(module, inp, out):
                # handle nested outputs
                t = out
                if not isinstance(out, torch.Tensor):
                    if isinstance(out, (list, tuple)) and out:
                        for o in out:
                            if isinstance(o, torch.Tensor):
                                t = o
                                break
                self._features[idx] = t
            return hook
        layers = self.yolo_model.model
        for layer_idx in self.backbone_feature_indices:
            if layer_idx < len(layers):
                self._hooks.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
    
    def _create_masks(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Create attention masks for features.
        For now, creates dummy masks (all valid), but could be enhanced
        to handle padding or invalid regions.
        
        Args:
            features: List of feature tensors
            
        Returns:
            List of boolean mask tensors
        """
        masks = []
        for f in feats:
            B, C, H, W = f.shape
            masks.append(torch.zeros((B, H, W), dtype=torch.bool, device=f.device))
        return masks
    
    def extract_backbone_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features from YOLO backbone.
        Also caches YOLO predictions to avoid a second forward.
        """
        # Clear previous cached features
        self._features.clear()
        self._last_predictions = None
        
        # Run forward pass through YOLO model to capture intermediate features and predictions
        with torch.no_grad():
            self._last_predictions = self.yolo_model(x)
        
        # Extract features at specified layer indices
        backbone_features = []
        for layer_idx in self.backbone_feature_indices:
            if layer_idx in self._features:
                backbone_features.append(self._features[layer_idx])
        
        return backbone_features
    
    def get_object_predictions(self, x: torch.Tensor) -> Any:
        """
        Return YOLO detections unchanged (as produced by the underlying model).
        Uses cached predictions from the most recent forward if available.
        """
        if self._last_predictions is not None:
            return self._last_predictions
        with torch.no_grad():
            return self.yolo_model(x)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[NestedTensor], List[torch.Tensor], Any]:
        """
        Forward pass to extract features and predictions in POET format.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            features: List of NestedTensor objects containing multi-scale features
            pos: List of positional encoding tensors (placeholder - all zeros for now)
            predictions: YOLO detections unchanged
        """
        # Extract backbone features
        backbone_features = self.extract_backbone_features(x)
        
        # Project features to consistent dimension
        projected_features = []
        for i, feat in enumerate(backbone_features):
            if i < len(self.input_proj):
                proj_feat = self.input_proj[i](feat)
                projected_features.append(proj_feat)
        
        # Generate additional feature levels if needed
        if len(projected_features) < self.num_feature_levels:
            for i in range(len(projected_features), self.num_feature_levels):
                if projected_features:
                    # Always use last projected feature (hidden_dim channels)
                    src = self.input_proj[i](projected_features[-1])
                else:
                    # Extremely unlikely, but if no backbone features captured, create from input via conv
                    # Here we just up-sample x to hidden_dim then downsample using the additional conv
                    b, _, h, w = x.shape
                    device = x.device
                    temp = torch.zeros((b, self.hidden_dim, h, w), device=device, dtype=x.dtype)
                    src = self.input_proj[i](temp)
                projected_features.append(src)
        
        # Create masks for features
        masks = self._create_masks(projected_features)
        
        # Create NestedTensor objects
        features = []
        for feat, mask in zip(projected_features, masks):
            features.append(NestedTensor(feat, mask))
        
        # Create placeholder positional encodings (all zeros for now)
        # In a full implementation, these would be proper sinusoidal positional encodings
        pos = []
        for feat in projected_features:
            B, C, H, W = feat.shape
            pos_embed = torch.zeros_like(feat)
            pos.append(pos_embed)
        
        # Get object predictions
        predictions = self.get_object_predictions(x)
        
        return features, pos, predictions
    
    def __del__(self):
        for h in getattr(self, '_hooks', []) or []:
            try:
                h.remove()
            except Exception:
                pass


class YOLOPoETBackbone(nn.Module):
    """
    Wrapper class that makes YOLO v11 compatible with POET framework.
    This class mimics the interface expected by POET's backbone.
    """
    
    def __init__(self, yolo_model: nn.Module, hidden_dim: int = 256, num_feature_levels: int = 3):
        """
        Initialize YOLO-POET backbone wrapper.
        
        Args:
            yolo_model: YOLO v11 model instance
            hidden_dim: Feature projection dimension
            num_feature_levels: Number of feature pyramid levels
        """
        super().__init__()
        
        self.adapter = PoETAdapter(yolo_model, hidden_dim, num_feature_levels)
        self.strides = self.adapter.strides
        self.num_channels = self.adapter.num_channels
        self.train_backbone = False  # Typically frozen for POET training
        
    def forward(self, tensor_list: NestedTensor) -> Tuple[List[NestedTensor], List[torch.Tensor], Any]:
        """
        Forward pass compatible with POET backbone interface.
        
        Args:
            tensor_list: NestedTensor containing input images and masks
            
        Returns:
            features: Multi-scale feature maps as NestedTensors
            pos: Positional encoding tensors
            predictions: Object detection predictions
        """
        # Extract input tensor from NestedTensor
        x = tensor_list.tensors
        
        # Forward through adapter
        features, pos, predictions = self.adapter(x)
        
        return features, pos, predictions


def create_yolo_poet_model(yolo_model_path: str, 
                          hidden_dim: int = 256, 
                          num_feature_levels: int = 3) -> YOLOPoETBackbone:
    """
    Factory function to create a YOLO-POET compatible model.
    
    Args:
        yolo_model_path: Path to YOLO v11 model weights
        hidden_dim: Feature projection dimension
        num_feature_levels: Number of feature pyramid levels
        
    Returns:
        YOLOPoETBackbone instance ready for POET integration
    """
    from ultralytics import YOLO
    
    # Load YOLO model
    yolo_model = YOLO(yolo_model_path)
    
    # Create POET-compatible backbone
    poet_backbone = YOLOPoETBackbone(yolo_model.model, hidden_dim, num_feature_levels)
    
    return poet_backbone
