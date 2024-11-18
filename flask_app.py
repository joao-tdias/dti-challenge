from flask import Flask, request, jsonify
import os
import sys
import numpy as np
from loguru import logger

sys.path.append(os.path.dirname(__file__))

from app.src.data_functions import load_data
from app.src.train_functions import train_and_evaluate_model, train_final_model
from app.src.prediction import predict_new_data
from app.constants.constants import LOGS_PATH, LOGS_LEVEL, LOGS_LEVEL_TERMINAL

app = Flask(__name__)

FILES = os.path.join(sys.path[-1], 'data')

logger.remove()
logger.add(sys.stderr, level=LOGS_LEVEL_TERMINAL)
logger.add(LOGS_PATH,
           level=LOGS_LEVEL,
           mode='a',
           rotation='5 MB'
           )

global_model = None
global_scaler = None

@app.route('/', methods=['GET', 'POST'])
def root():
    return jsonify({'status':'Aplicação online'})

@app.route('/train', methods=['POST'])
def train():
    data_path = os.path.join(FILES, 'breast-cancer-wisconsin.data')
    test_size = float(request.form.get('test_size', 0.3))
    eta0 = float(request.form.get('eta0', 0.1))
    random_seed_start = 1
    random_seed_end = 20

    X, y = load_data(data_path)
    random_seed_array = range(random_seed_start, random_seed_end + 1)

    accuracy_array, precision_array = train_and_evaluate_model(
        X, y, test_size, eta0, random_seed_array
    )

    accuracy_percent = [round(acc * 100, 2) for acc in accuracy_array]
    precision_percent = [float(round(prec * 100, 2)) for prec in precision_array]

    logger.info('Obtained accuracies: {0}'.format(accuracy_percent))
    logger.info('Obtained precisions: {0}'.format(precision_percent))
    logger.info('Standard deviation of accuracy: {0:.2f}'.format(np.std(accuracy_array)))
    logger.info('Standard deviation of precision: {0:.2f}'.format(np.std(precision_array)))

    global global_model
    global global_scaler
    global_model, global_scaler = train_final_model(X, y, eta0)

    return jsonify({
        'accuracy': accuracy_percent,
        'precision': precision_percent,
        'accuracy_std_dev': round(np.std(accuracy_array), 2),
        'precision_std_dev': round(np.std(precision_array), 2)
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'data' not in data:
        return jsonify({'error': 'No data provided for prediction'}), 400

    if global_model is None or global_scaler is None:
        return jsonify({'error': 'Model not trained. Please train the model first.'}), 400

    try:
        prediction = predict_new_data(global_model, global_scaler, data['data'])
        logger.success(f"Predicted: {prediction}")
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/run', methods=['GET'])
def run():
    data_path = os.path.join(FILES, 'breast-cancer-wisconsin.data')
    test_size = 0.3
    eta0 = 0.1
    random_seed_start = 1
    random_seed_end = 20

    X, y = load_data(data_path)
    random_seed_array = range(random_seed_start, random_seed_end + 1)

    accuracy_array, precision_array = train_and_evaluate_model(
        X, y, test_size, eta0, random_seed_array
    )

    accuracy_percent = [round(acc * 100, 2) for acc in accuracy_array]
    precision_percent = [float(round(prec * 100, 2)) for prec in precision_array]

    logger.info('Obtained accuracies: {0}'.format(accuracy_percent))
    logger.info('Obtained precisions: {0}'.format(precision_percent))
    logger.info('Standard deviation of accuracy: {0:.2f}'.format(np.std(accuracy_array)))
    logger.info('Standard deviation of precision: {0:.2f}'.format(np.std(precision_array)))

    global global_model
    global global_scaler
    global_model, global_scaler = train_final_model(X, y, eta0)

    data_example = [[999999, 5, 10, 10, 3, 7, 3, 8, 10, 2]]
    pred = predict_new_data(global_model, global_scaler, data_example)
    logger.success(f"Predicted: {pred}")

    return jsonify({
        'accuracy': accuracy_percent,
        'precision': precision_percent,
        'accuracy_std_dev': round(np.std(accuracy_array), 2),
        'precision_std_dev': round(np.std(precision_array), 2),
        'prediction_example': pred.tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
