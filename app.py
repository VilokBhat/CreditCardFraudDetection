'''
from flask import Flask, request, jsonify, render_template
import joblib
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# Load the trained model
try:
    logistic_regression_model = joblib.load('models/LogisticRegression_model.pkl')
    random_forest_model = joblib.load('models/RandomForest_model.pkl')
    #xgboost_model = joblib.load('models/XGBoost_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
except Exception as e:
    print(f"Error loading models or scaler: {e}")

# define the risk category based on the fraud probability
def risk_category(probability):
    if probability < 0.1:
        return 'Low'
    elif probability < 0.5:
        return 'Medium'
    else:
        return 'High'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get features from the form
    input_data = request.form['features']  # features except Time and Amount
    time = float(request.form['time'])
    amount = float(request.form['amount'])
    model_choice = request.form['model']

    # Convert the features into a list of floats
    features = [float(x) for x in input_data.split(',')]

    # Scale Time and Amount separately
    scaled_time = scaler.transform([[time]])[0][0]  # Scaling Time
    scaled_amount = scaler.transform([[amount]])[0][0]  # Scaling Amount

    # Append the scaled Time and Amount to the features
    features.append(scaled_time)
    features.append(scaled_amount)

    if model_choice == 'Logistic Regression':
        model = logistic_regression_model
        model_name = 'Logistic Regression'
    elif model_choice == 'Random Forest':
        model = random_forest_model
        model_name = 'Random Forest'
    else:
        #model = xgboost_model
        model_name = 'XGBoost'

    # Make prediction using the trained model
    prediction = model.predict([features])[0]
    fraud_probability = model.predict_proba([features])[0][1]
    prediction_text = 'Fraud' if prediction == 1 else 'Not Fraud'

    return render_template('index.html', 
                           prediction_text=f'Prediction: {prediction_text} (Probability: {fraud_probability:.2f})')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Use PORT from environment or default to 8080
    app.run(host="0.0.0.0",port = 8080)#port=port, debug=True  # Run the app
'''    
from flask import Flask, request, jsonify, render_template
import joblib
import os
#from dotenv import load_dotenv

# Load environment variables from .env
#load_dotenv()

app = Flask(__name__)

# Define the models and their accuracies
models_info = {
    'LogisticRegression_model1500.pkl': 0.9459,
    'RandomForest_model1500.pkl': 0.9742,
    'XGBoost_model1500.pkl': 0.9717,
    'LogisticRegression_model.pkl': 0.9491,
    'RandomForest_model.pkl': 0.9999,
    'XGBoost_model.pkl': 0.9997,
    'LogisticRegression_model2500.pkl': 0.9420,
    'RandomForest_model2500.pkl': 0.9850,
    'XGBoost_model2500.pkl': 0.9850,
    'LogisticRegression_model3000.pkl': 0.9388,
    'RandomForest_model3000.pkl': 0.9867,
    'XGBoost_model3000.pkl': 0.9863,
    'LogisticRegression_model4000.pkl': 0.9444,
    'RandomForest_model4000.pkl': 0.9944,
    'XGBoost_model4000.pkl': 0.9929,
    'LogisticRegression_model5000.pkl': 0.9478,
    'RandomForest_model5000.pkl': 0.9950,
    'XGBoost_model5000.pkl': 0.9938,
    'LogisticRegression_model7000.pkl': 0.9454,
    'RandomForest_model7000.pkl': 0.9980,
    'XGBoost_model7000.pkl': 0.9964,
    'LogisticRegression_model10000.pkl': 0.9478,
    'RandomForest_model10000.pkl': 0.9990,
    'XGBoost_model10000.pkl': 0.9971
}

# Select the model with the highest accuracy
best_model_file = max(models_info, key=models_info.get)
best_model_accuracy = models_info[best_model_file]

# Load the best model
try:
    best_model = joblib.load(f'models/{best_model_file}')
    scaler = joblib.load('models/scaler.pkl')
except Exception as e:
    print(f"Error loading model or scaler: {e}")

# Define the risk category based on the fraud probability
def risk_category(probability):
    if probability < 0.1:
        return 'Low'
    elif probability < 0.5:
        return 'Medium'
    else:
        return 'High'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get features from the form
    input_data = request.form['features']  # features except Time and Amount
    time = float(request.form['time'])
    amount = float(request.form['amount'])

    # Convert the features into a list of floats
    features = [float(x) for x in input_data.split(',')]

    # Scale Time and Amount separately
    scaled_time = scaler.transform([[time]])[0][0]  # Scaling Time
    scaled_amount = scaler.transform([[amount]])[0][0]  # Scaling Amount

    # Append the scaled Time and Amount to the features
    features.append(scaled_time)
    features.append(scaled_amount)

    # Make prediction using the best model
    prediction = best_model.predict([features])[0]
    fraud_probability = best_model.predict_proba([features])[0][1]
    prediction_text = 'Fraud' if prediction == 1 else 'Not Fraud'

    return render_template('index.html', 
                           prediction_text=f'Prediction: {prediction_text} (Probability: {fraud_probability:.2f})')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Use PORT from environment or default to 8080
    app.run(host="0.0.0.0", port=8080)  # Run the app
'''
import os
import joblib
from flask import Flask, request, jsonify, render_template
#from dotenv import load_dotenv

# Load environment variables from .env
#load_dotenv()

app = Flask(__name__)

# Define the path to the models and mlruns directories
models_dir = 'models'
mlruns_dir = 'mlruns'

# Load the models and their accuracies from the mlruns directory
models_info = {}
for root, dirs, files in os.walk(mlruns_dir):
    for file in files:
        if file == 'accuracy':  # Assuming accuracy is stored in a file named 'accuracy'
            accuracy_path = os.path.join(root, file)
            with open(accuracy_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        model_id, accuracy, _ = parts
                        model_path = os.path.join(models_dir, f'{model_id}.pkl')
                        models_info[model_path] = float(accuracy)

# Select the model with the highest accuracy
best_model_file = max(models_info, key=models_info.get)
best_model_accuracy = models_info[best_model_file]

# Load the best model and scaler
try:
    best_model = joblib.load(best_model_file)
    scaler = joblib.load('models/scaler.pkl')
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    best_model = None
    scaler = None

# Define the risk category based on the fraud probability
def risk_category(probability):
    if probability < 0.1:
        return 'Low'
    elif probability < 0.5:
        return 'Medium'
    else:
        return 'High'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if best_model is None or scaler is None:
        return render_template('index.html', prediction_text='Error: Model or scaler not loaded.')

    # Get features from the form
    input_data = request.form['features']  # features except Time and Amount
    time = float(request.form['time'])
    amount = float(request.form['amount'])

    # Convert the features into a list of floats
    features = [float(x) for x in input_data.split(',')]

    # Scale Time and Amount separately
    scaled_time = scaler.transform([[time]])[0][0]  # Scaling Time
    scaled_amount = scaler.transform([[amount]])[0][0]  # Scaling Amount

    # Append the scaled Time and Amount to the features
    features.append(scaled_time)
    features.append(scaled_amount)

    # Make prediction using the best model
    prediction = best_model.predict([features])[0]
    fraud_probability = best_model.predict_proba([features])[0][1]
    prediction_text = 'Fraud' if prediction == 1 else 'Not Fraud'

    return render_template('index.html', 
                           prediction_text=f'Prediction: {prediction_text} (Probability: {fraud_probability:.2f})')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Use PORT from environment or default to 8080
    app.run(host="0.0.0.0", port=8080)  # Run the app
    '''