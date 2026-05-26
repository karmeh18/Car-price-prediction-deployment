import os
import sys
import warnings
warnings.filterwarnings('ignore')

from src.exception import Custom_Exception
from src.logger import logging
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class DataIngestionConfig:
    # Get absolute path for cross-platform compatibility
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_data_path = os.path.join(project_root, 'artifacts', 'train.csv')
    test_data_path = os.path.join(project_root, 'artifacts', 'test.csv')
    raw_data_path = os.path.join(project_root, 'artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered into Data Ingestion stage, file is ready to import")
        try:
            df=pd.read_csv(r'notebook\car_data.csv')
            logging.info('Raw Data has been loaded and read as DataFrame')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("Train-Test split has been initiated")
            train_data,test_data=train_test_split(df,test_size=0.3,random_state=42)

            logging.info("Training Data and Test Data will be exported to respective path destination")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            train_data.to_csv(self.ingestion_config.train_data_path,header=True,index=False)

            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Data Splitting has been completed and csv files have been exported")

            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path)
        except Exception as e:
            raise Custom_Exception(e,sys)
        

if __name__=="__main__":
    import os
    import hashlib
    
    print("\n" + "="*60)
    print("🔍 Checking for existing model and data changes...")
    print("="*60)
    
    # Get absolute paths for cross-platform compatibility
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    source_data_path = os.path.join(project_root, r'notebook\car_data.csv')
    model_path = os.path.join(project_root, 'artifacts', 'model.pkl')
    data_hash_file = os.path.join(project_root, 'artifacts', '.data_hash')
    
    # Check if model exists
    model_exists = os.path.exists(model_path)
    
    # Function to get file checksum (detect changes)
    def get_file_hash(filepath):
        """Calculate MD5 hash of file"""
        try:
            hash_md5 = hashlib.md5()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return None
    
    # Get current data hash
    current_data_hash = get_file_hash(source_data_path)
    
    # Get previous data hash (if exists)
    previous_data_hash = None
    if os.path.exists(data_hash_file):
        try:
            with open(data_hash_file, 'r') as f:
                previous_data_hash = f.read().strip()
        except:
            pass
    
    # Check if data has changed
    data_changed = current_data_hash != previous_data_hash
    
    if model_exists:
        print(f"✓ Model exists: {model_path}")
        
        if data_changed:
            print("\n⚠️  ⚠️  DATA HAS CHANGED! ⚠️  ⚠️")
            print("─" * 60)
            print("New data detected in notebook/car_data.csv")
            print("The existing model may be outdated!")
            print("─" * 60)
            retrain = input("\n⚠️  IMPORTANT: Do you want to RETRAIN with new data? (yes/no): ")
            
            if retrain.lower() != 'yes' and retrain.lower() != 'y':
                print("\n❌ WARNING: Using OLD model with NEW data!")
                print("   This may give inaccurate predictions!")
                print("   Recommend: Run again and select 'yes' to retrain")
                use_old = input("\nContinue anyway? (yes/no): ")
                if use_old.lower() != 'yes' and use_old.lower() != 'y':
                    print("✓ Exiting. Please retrain the model.")
                    sys.exit(0)
            else:
                print("⏳ Proceeding with retraining using new data...")
        else:
            print("✓ Data unchanged")
            retrain = input("\nDo you want to retrain the model? (yes/no): ")
            
            if retrain.lower() != 'yes' and retrain.lower() != 'y':
                print("\n✓ Using existing model")
                print(f"  Model path: {model_path}")
                print("  To use model for predictions, run: python app.py")
                sys.exit(0)  # Exit without retraining
            else:
                print("⏳ Proceeding with retraining...")
    else:
        print("✗ No existing model found")
        print("  Will train a new model...")
    
    # Proceed with pipeline
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)
    
    trans=DataTransformation()
    train_arr,test_arr,feature_names=trans.initiate_data_transformation(train_path=train_data_path,test_path=test_data_path)

    # Optional: Model Comparison
    print("\n" + "="*60)
    compare_models = input("Compare multiple models (RandomForest vs GB vs XGBoost vs CatBoost)? (yes/no): ")
    if compare_models.lower() == 'yes' or compare_models.lower() == 'y':
        from src.components.model_comparison import ModelComparison
        
        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
        
        model_comparator = ModelComparison()
        best_model, best_name, report = model_comparator.run_full_comparison(
            X_train, y_train, X_test, y_test, cv_folds=5, feature_names=feature_names
        )
        
        print("\n💾 Best model saved to artifacts/model.pkl")
        print("📊 Comparison reports saved to artifacts/reports/")
        print("📄 HTML Report saved to artifacts/model_comparison_report.html")
        
        # Save data hash for future reference
        os.makedirs('artifacts', exist_ok=True)
        with open(data_hash_file, 'w') as f:
            f.write(current_data_hash)
        print(f"✓ Data hash saved for future change detection")
    else:
        # Standard training
        model_trainer=ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr,test_arr)
        
        # Save data hash
        os.makedirs('artifacts', exist_ok=True)
        with open(data_hash_file, 'w') as f:
            f.write(current_data_hash)