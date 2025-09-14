#!/usr/bin/env python3
"""
Visualize YOLO v11 + POETAdapter outputs on an image using OpenCV.
- Draw YOLO detections
- Overlay multi-scale feature map heatmaps

Usage:
  python examples/visualize_poet_cv2.py --image path/to/img.jpg [--weights yolo11n.pt] [--levels 3] [--hidden 256] [--show]

Notes:
- The adapter returns features; YOLO detections are obtained from the high-level API for easy box decoding.
- Image is resized to 640x640 for a simple end-to-end demo.
"""
import argparse
import os
import sys

import numpy as np
import torch

try:
    import cv2
except Exception as e:
    print("OpenCV (cv2) is required. Install with: pip install opencv-python")
    raise

# Prioritize local repo over site-packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import YOLO
from ultralytics.nn.modules.poet_adapter import PoETAdapter


def mk_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def to_tensor_640(image_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)  # 1x3x640x640


def draw_yolo_boxes(image_bgr: np.ndarray, results) -> np.ndarray:
    img = image_bgr.copy()
    if not results:
        return img
    res = results[0]
    if getattr(res, 'boxes', None) is None:
        return img
    boxes = res.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy() if getattr(boxes, 'conf', None) is not None else np.ones(len(xyxy))
    cls = boxes.cls.cpu().numpy().astype(int) if getattr(boxes, 'cls', None) is not None else np.zeros(len(xyxy), dtype=int)
    names = res.names if hasattr(res, 'names') else {i: str(i) for i in np.unique(cls)}
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        p1 = (int(x1), int(y1)); p2 = (int(x2), int(y2))
        color = (0, 255, 0)
        cv2.rectangle(img, p1, p2, color, 2)
        label = f"{names.get(k, k)} {c:.2f}"
        cv2.putText(img, label, (p1[0], max(0, p1[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def feature_heatmap_overlay(image_bgr: np.ndarray, feat: torch.Tensor) -> np.ndarray:
    # feat: 1xCxHxW
    fmap = feat[0]  # CxHxW
    # aggregate using L2 norm across channels
    agg = torch.norm(fmap, p=2, dim=0)
    agg = (agg - agg.min()) / (agg.max() - agg.min() + 1e-6)
    heat = (agg.cpu().numpy() * 255).astype(np.uint8)  # HxW
    heat = cv2.resize(heat, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    heat_c = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 0.6, heat_c, 0.4, 0)
    return overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, help='Path to input image')
    ap.add_argument('--weights', default='yolo11n.pt', help='YOLOv11 weights')
    ap.add_argument('--levels', type=int, default=3, help='Number of feature levels')
    ap.add_argument('--hidden', type=int, default=256, help='Hidden dim for features')
    ap.add_argument('--show', action='store_true', help='Use cv2.imshow; saves to files if not set or if GUI unavailable')
    ap.add_argument('--outdir', default=os.path.join('examples', 'outputs'), help='Directory to save outputs')
    args = ap.parse_args()

    outdir = mk_outdir(args.outdir)

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    # For this demo, operate on a 640x640 resized copy
    img_bgr_640 = cv2.resize(img_bgr, (640, 640), interpolation=cv2.INTER_LINEAR)

    # Load models
    yolo = YOLO(args.weights)
    adapter = PoETAdapter(yolo.model, hidden_dim=args.hidden, num_feature_levels=args.levels)

    # Prepare tensor and run adapter
    x = to_tensor_640(img_bgr_640)
    with torch.no_grad():
        features, pos, _ = adapter(x)

    # Get YOLO detections via high-level API for easy box decoding
    results = yolo(img_bgr_640, imgsz=640, verbose=False)

    # Draw detections
    det_img = draw_yolo_boxes(img_bgr_640, results)

    # Overlays for each feature level
    overlays = []
    for i, nt in enumerate(features):
        t, _ = nt.decompose()
        ov = feature_heatmap_overlay(img_bgr_640, t)
        overlays.append((i, ov))

    # Try to show via OpenCV or save if GUI not available
    showed = False
    if args.show:
        try:
            cv2.imshow('Detections', det_img)
            for i, ov in overlays:
                cv2.imshow(f'Feature L{i}', ov)
            print('Press any key in the image window to close...')
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            showed = True
        except Exception as e:
            print(f"cv2.imshow failed ({e}); saving to files instead.")

    # Save outputs
    if not showed:
        det_path = os.path.join(outdir, 'detections.jpg')
        cv2.imwrite(det_path, det_img)
        saved = [det_path]
        for i, ov in overlays:
            p = os.path.join(outdir, f'feature_overlay_L{i}.jpg')
            cv2.imwrite(p, ov)
            saved.append(p)
        print('Saved:', *saved, sep='\n  ')


if __name__ == '__main__':
    main()
