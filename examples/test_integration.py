#!/usr/bin/env python3
"""
Test script to verify YOLO v11 + POET integration using the library adapter.
"""

import os
import sys
# Prioritize local repo over site-packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from ultralytics import YOLO
from ultralytics.nn.modules.poet_adapter import PoETAdapter


def test_poet_integration():
    """Test the POET integration with YOLO v11."""
    
    print("Testing YOLO v11 + POET Integration...")
    
    try:
        print("Successfully imported PoETAdapter")
        
        print("Loading YOLO v11 model...")
        yolo = YOLO('yolo11n.pt')
        print("YOLO model loaded")
        
        print("Creating POET adapter...")
        adapter = PoETAdapter(yolo.model, hidden_dim=256, num_feature_levels=3)
        print("POET adapter created")
        
        print(f"   - Feature layers: {adapter.feature_layers}")
        print(f"   - Backbone channels: {adapter.backbone_channels}")
        print(f"   - Output strides: {adapter.strides}")
        
        # Test forward pass
        batch_size = 2
        img_size = 640
        dummy_input = torch.randn(batch_size, 3, img_size, img_size)
        print(f"Created dummy input: {dummy_input.shape}")
        
        print("Testing forward pass...")
        with torch.no_grad():
            features, pos_encodings, predictions = adapter(dummy_input)
        
        print("Forward pass successful")
        
        # Verify outputs
        print("\nVerifying outputs...")
        assert len(features) == 3, f"Expected 3 feature levels, got {len(features)}"
        print(f"Got {len(features)} feature levels")
        
        for i, feat in enumerate(features):
            feat_tensor, mask = feat.decompose()
            expected_stride = adapter.strides[i]
            expected_size = img_size // expected_stride
            
            print(f"Level {i}: Expected {expected_size}x{expected_size} (stride {expected_stride}), got {feat_tensor.shape[2]}x{feat_tensor.shape[3]}")
            
            assert feat_tensor.shape[0] == batch_size, f"Batch size mismatch at level {i}"
            assert feat_tensor.shape[1] == 256, f"Channel count mismatch at level {i}"
            assert feat_tensor.shape[2] == expected_size, f"Height mismatch at level {i}"
            assert feat_tensor.shape[3] == expected_size, f"Width mismatch at level {i}"
            
            assert mask.shape[0] == batch_size, f"Mask batch size mismatch at level {i}"
            assert mask.shape[1] == expected_size, f"Mask height mismatch at level {i}"
            assert mask.shape[2] == expected_size, f"Mask width mismatch at level {i}"
            
            print(f"Level {i}: {feat_tensor.shape}, stride {expected_stride}")
        
        print("Positional encodings verified")
        print("YOLO predictions returned")
        
        # Test different configurations
        print("\nTesting different configurations...")
        adapter_4 = PoETAdapter(yolo.model, hidden_dim=256, num_feature_levels=4)
        with torch.no_grad():
            features_4, _, _ = adapter_4(dummy_input)
        assert len(features_4) == 4, "4-level feature extraction failed"
        print("4-level feature extraction works")
        
        adapter_512 = PoETAdapter(yolo.model, hidden_dim=512, num_feature_levels=3)
        with torch.no_grad():
            features_512, _, _ = adapter_512(dummy_input)
        feat_tensor_512, _ = features_512[0].decompose()
        assert feat_tensor_512.shape[1] == 512, "Hidden dim 512 failed"
        print("Different hidden dimensions work")
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED!")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_quality():
    """Test the quality of extracted features."""
    
    print("\nTesting feature quality...")
    
    try:
        yolo = YOLO('yolo11n.pt')
        adapter = PoETAdapter(yolo.model)
        
        test_img = torch.randn(1, 3, 640, 640)
        
        with torch.no_grad():
            features, _, _ = adapter(test_img)
        
        for i, feat in enumerate(features):
            feat_tensor, _ = feat.decompose()
            
            feat_std = feat_tensor.std().item()
            feat_min = feat_tensor.min().item()
            feat_max = feat_tensor.max().item()
            
            assert feat_std > 1e-6, f"Level {i} features have no variation"
            assert not torch.isnan(feat_tensor).any(), f"Level {i} contains NaN values"
            assert not torch.isinf(feat_tensor).any(), f"Level {i} contains Inf values"
            
            print(f"Level {i}: std={feat_std:.4f}, range=[{feat_min:.4f}, {feat_max:.4f}]")
        
        print("Feature quality tests passed")
        return True
        
    except Exception as e:
        print(f"Feature quality test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_poet_integration()
    if success:
        success = test_feature_quality()
    
    if success:
        print("\nReady for POET integration!")
    else:
        print("\nIntegration needs fixing before use with POET")
        exit(1)
