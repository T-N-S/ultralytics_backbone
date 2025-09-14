# YOLO v11 + POET Integration

```python
from ultralytics import YOLO
from ultralytics.nn.modules.poet_adapter import PoETAdapter

# Load YOLO and create adapter
yolo = YOLO('yolo11n.pt')
adapter = PoETAdapter(yolo.model, hidden_dim=256, num_feature_levels=3)

# Extract features
features, pos, predictions = adapter(image_tensor)
```

## What You Get

- Multi-scale features (auto-detected strides e.g., 8/16/32)
- Consistent hidden_dim channels (default 256)
- NestedTensor format with masks
- Original YOLO detections preserved

Notes:
- Backbone feature layers and strides are detected automatically. You can override with `backbone_feature_indices=[...]`.
- `num_feature_levels` can exceed backbone taps; extra levels are generated from the last projected map.
- Positional encodings are zeros by default. Add your own PE if your POET setup requires it.

## Files

- `examples/test_integration.py` - Single concise test

## Testing

```bash
python examples/test_integration.py
```
