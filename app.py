from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Pipeline.predict_pipeline import CustomData,PredictPipeline

app=Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="GET":
        return render_template('index.html')
    else:
        try:
            data=CustomData(
                Company=request.form.get('Company'),
                Engine=request.form.get('Engine'),
                Transmission=request.form.get('Transmission'),
                Color=request.form.get('Color'),
                Body_Style=request.form.get('Body_Style'),
                Dealer_Region=request.form.get('Dealer_Region'),
            )
            pred_df=data.get_data_as_dataframe()
            print(pred_df)
            predict_pipeline=PredictPipeline()
            results=predict_pipeline.predict(pred_df)
            return render_template('index.html', results=str(np.round(results[0],2)))
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return render_template('index.html', results=str(e))
    
if __name__=='__main__':
    app.run(host='0.0.0.0',port=9090,debug=True)
