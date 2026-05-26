"""
Legacy retrain script - Now integrated into data_ingestion.py pipeline
Use this for standalone retraining or as a fallback

For integrated pipeline, run:
    python src/components/data_ingestion.py
    
And select 'yes' when prompted for hyperparameter optimization
Then select 'yes' when prompted to retrain on full dataset
"""
import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def standalone_retrain():
    """Standalone retraining without hyperparameter optimization"""
    print("=" * 60)
    print("🔄 Standalone Model Retraining")
    print("(For integrated pipeline, run: python src/components/data_ingestion.py)")
    print("=" * 60)
    
    # Step 1: Data Ingestion
    print("\n[1/3] Data Ingestion in progress...")
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    print(f"✓ Data ingestion completed")
    print(f"  - Train data: {train_data_path}")
    print(f"  - Test data: {test_data_path}")
    
    # Step 2: Data Transformation
    print("\n[2/3] Data Transformation in progress...")
    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.initiate_data_transformation(
        train_path=train_data_path,
        test_path=test_data_path
    )
    print(f"✓ Data transformation completed")
    print(f"  - Preprocessor saved: artifacts/preprocessor.pkl")
    
    # Step 3: Model Training (without hyperparameter optimization)
    print("\n[3/3] Model Training in progress...")
    model_trainer = ModelTrainer()
    mape_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"✓ Model training completed")
    print(f"  - Model saved: artifacts/model.pkl")
    print(f"  - MAPE Score: {mape_score:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ Retraining completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run Flask app: python app.py")
    print("  2. Or for optimized params, run: python src/components/data_ingestion.py")

if __name__ == "__main__":
    try:
        standalone_retrain()
    except Exception as e:
        print(f"\n✗ Error during retraining: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
