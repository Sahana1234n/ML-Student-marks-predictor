from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.utils import load_object
import os

# Initialize Flask app
application = Flask(__name__)
app = application

# Route for home page + prediction
@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Collect data from form
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=request.form.get('reading_score'),
            writing_score=request.form.get('writing_score')
        )
        pred_df = data.get_data_as_data_frame()

        # Prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Return home.html with results
        return render_template('home.html', results=round(results[0], 2))

    # GET request → show form
    return render_template('home.html')


# Run app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
