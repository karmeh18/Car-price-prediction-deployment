import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.exception import Custom_Exception
from src.logger import logging

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.utils import save_object, calculate_mape

@dataclass
class DataTransformation:
    try:
        def get_data_transformation_object(self):
            # Get absolute path for cross-platform compatibility
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.get_data_transformation_config = os.path.join(project_root, 'artifacts', 'preprocessor.pkl')
            cat_col=['Company', 'Engine', 'Transmission', 'Color', 'Body_Style', 'Dealer_Region']
            
            cat_pipeline=Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy='most_frequent')),
                    ("One-Hot-Encoder",OneHotEncoder()),
                    ('StandardScaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info('Pipeline has been initiated for Categorical Columns {}'.format(cat_col))

            preprocessor=ColumnTransformer(
                [ ('cat_pipeline',cat_pipeline,cat_col)
                ]
            )
            logging.info("Preprocessor has been completed")

            return preprocessor
    except Exception as e:
        raise Custom_Exception(e,sys)

    try:
        def initiate_data_transformation(self,train_path,test_path):
            scaler=StandardScaler()
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)            

            logging.info("Training Data and Test Data has been loaded")
            logging.info("Data Transformation and Preprocessing Initiated")

            train_df["Engine"]=np.where(train_df["Engine"]=="DoubleÂ Overhead Camshaft","Double Overhead Camshaft",train_df["Engine"])
            train_df["Engine"]=np.where(train_df["Engine"]=="Overhead Camshaft","Single Overhead Camshaft",train_df["Engine"])
            train_df.rename(columns={"Price ($)":"Price"},inplace=True)
            train_df.rename(columns={"Body Style":"Body_Style"},inplace=True)
            logging.info("Data Preprocessing of Training Data has been completed")

            test_df["Engine"]=np.where(test_df["Engine"]=="DoubleÂ Overhead Camshaft","Double Overhead Camshaft",test_df["Engine"])
            test_df["Engine"]=np.where(test_df["Engine"]=="Overhead Camshaft","Single Overhead Camshaft",test_df["Engine"])
            test_df.rename(columns={"Price ($)":"Price"},inplace=True)
            test_df.rename(columns={"Body Style":"Body_Style"},inplace=True)
            logging.info('Data Preprocessing of Test Data has been completed')

            X_train = train_df[['Company', 'Engine', 'Transmission', 'Color', 'Body_Style', 'Dealer_Region']]
            y_train = train_df["Price"]

            X_test = test_df[['Company', 'Engine', 'Transmission', 'Color', 'Body_Style', 'Dealer_Region']]
            y_test = test_df["Price"]
            logging.info("Training Data and Testing Data has been seperated into X_train,y_train,X_test,y_test datasets")

            logging.info("Applying Preprocessor Object on Training and Test Datasets")
            preprocessor_obj=self.get_data_transformation_object()
            
            input_feature_train_arr=preprocessor_obj.fit_transform(X_train)
            input_feature_test_arr=preprocessor_obj.fit_transform(X_test)
            logging.info("Preprocessor Object has been implemented")

            train_arr=np.c_[input_feature_train_arr.toarray(),np.array(y_train)]
            test_arr=np.c_[input_feature_test_arr.toarray(),np.array(y_test)]
            logging.info("Preprocessed Independent variables have been concatenated with Dependent variable")

            save_object(file_path=self.get_data_transformation_config,obj=preprocessor_obj)
            
            # Get feature names after one-hot encoding
            try:
                feature_names = preprocessor_obj.get_feature_names_out()
            except:
                # Fallback if get_feature_names_out is not available
                feature_names = [f"Feature_{i}" for i in range(input_feature_train_arr.shape[1])]

            return train_arr,test_arr,feature_names
                
    except Exception as e:
        raise Custom_Exception(e,sys)        
    