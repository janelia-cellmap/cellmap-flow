#!/usr/bin/env python3
"""Simple test for dinov3_playground imports."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports from dinov3_playground...")

try:
    # Test individual module imports
    print("1. Testing dinov3_core...")
    from dinov3_playground.dinov3_core import enable_amp_inference
    print("   ✅ dinov3_core import successful")
    
    print("2. Testing data_processing...")
    from dinov3_playground.data_processing import sample_training_data
    print("   ✅ data_processing import successful")
    
    print("3. Testing models...")
    from dinov3_playground.models import ImprovedClassifier
    print("   ✅ models import successful")
    
    print("4. Testing model_training...")
    from dinov3_playground.model_training import balance_classes
    print("   ✅ model_training import successful")
    
    print("5. Testing visualization...")
    from dinov3_playground.visualization import plot_training_history
    print("   ✅ visualization import successful")
    
    print("6. Testing package-level imports...")
    from dinov3_playground import process, ImprovedClassifier
    print("   ✅ package-level imports successful")
    
    print("\n✅ ALL IMPORTS SUCCESSFUL!")
    print("The dinov3_playground package is working correctly.")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTo run the full examples:")
print("  cd dinov3_playground")
print("  python dinov3_finetune.py")
