"""
Quick test of model comparison functionality
"""
import sys
import os
sys.path.insert(0, r'd:\car_prediction_project')

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_comparison import ModelComparison
import numpy as np

# Step 1: Data Ingestion
print("=" * 60)
print("🔄 STEP 1: Data Ingestion")
print("=" * 60)
obj = DataIngestion()
train_data_path, test_data_path = obj.initiate_data_ingestion()
print(f"✓ Data loaded: train={train_data_path}, test={test_data_path}")

# Step 2: Data Transformation
print("\n" + "=" * 60)
print("🔄 STEP 2: Data Transformation")
print("=" * 60)
trans = DataTransformation()
train_arr, test_arr, feature_names = trans.initiate_data_transformation(
    train_path=train_data_path, 
    test_path=test_data_path
)
print(f"✓ Data transformed: train shape={train_arr.shape}, test shape={test_arr.shape}")
print(f"✓ Feature names: {len(feature_names)} features")

# Step 3: Model Comparison
print("\n" + "=" * 60)
print("🔄 STEP 3: Model Comparison")
print("=" * 60)

X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

model_comparator = ModelComparison()
best_model, best_name, report = model_comparator.run_full_comparison(
    X_train, y_train, X_test, y_test, cv_folds=5, feature_names=list(feature_names)
)

print(f"\n✓ Model comparison complete!")
print(f"  Best model: {best_name}")
print(f"  Report keys: {list(report.keys())}")

# Check artifacts
print("\n" + "=" * 60)
print("📁 Generated Artifacts")
print("=" * 60)
artifacts_dir = 'artifacts'
if os.path.exists(artifacts_dir):
    files = os.listdir(artifacts_dir)
    for f in sorted(files):
        fpath = os.path.join(artifacts_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath) / 1024
            print(f"✓ {f} ({size:.1f} KB)")
else:
    print("✗ No artifacts directory found")

print("\n" + "=" * 60)
print("✅ TEST COMPLETE")
print("=" * 60)
