import sys
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.exception import Custom_Exception
from src.logger import logging
from src.utils import load_obj

class PredictPipeline:
    def predict(self,feature):
        try:
            model_path=os.path.join("artifacts",'model.pkl')
            processor_path=os.path.join('artifacts','preprocessor.pkl')
            logging.info("Model and Preprocessor object has been loaded")

            model=load_obj(model_path)
            preprocessor=load_obj(processor_path)

            data_scaled=preprocessor.transform(feature)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise Custom_Exception(e,sys)

class CustomData:
    def __init__(self,
                 Company,
                 Engine,
                 Transmission,
                 Color,
                 Body_Style,
                 Dealer_Region
                 ):

        self.Company=Company
        self.Engine=Engine
        self.Transmission=Transmission
        self.Color=Color
        self.Body_Style=Body_Style
        self.Dealer_Region=Dealer_Region

    def get_data_as_dataframe(self):
        try:
            custom_data_input={
                'Company':[self.Company],
                'Engine':[self.Engine],
                'Transmission':[self.Transmission],
                'Color':[self.Color],
                'Body_Style':[self.Body_Style],
                'Dealer_Region':[self.Dealer_Region]
            }
            return pd.DataFrame(custom_data_input)
        except Exception as e:
            raise Custom_Exception(e,sys)