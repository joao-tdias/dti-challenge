import os
import sys
import numpy as np
from loguru import logger

sys.path.append(os.path.dirname(__file__))

from app.src.data_functions import load_data
from app.src.train_functions import train_and_evaluate_model, train_final_model
from app.src.prediction import predict_new_data
from app.constants.constants import LOGS_PATH, LOGS_LEVEL, LOGS_LEVEL_TERMINAL

FILES = os.path.join(os.path.join(sys.path[-1], 'app'),'data')

logger.remove()
logger.add(sys.stderr, level=LOGS_LEVEL_TERMINAL)
logger.add(LOGS_PATH,
           level=LOGS_LEVEL,
           mode='a',
           rotation='5 MB'
           )

def main():
    """
    Breast Cancer Classification with Perceptron
    """
    # Set default parameters
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

    # Train the final model on the entire dataset
    model, scaler = train_final_model(X, y, eta0)

    # Example prediction
    data = [[999999, 5, 10, 10, 3, 7, 3, 8, 10, 2]]
    pred = predict_new_data(model, scaler, data)
    logger.success(f"Predicted: {pred}")

if __name__ == '__main__':
    main()
