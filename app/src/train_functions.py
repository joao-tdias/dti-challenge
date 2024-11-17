from loguru import logger
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from constants.constants import LOGS_PATH, LOGS_LEVEL

logger.add(
    LOGS_PATH,
    level=LOGS_LEVEL,
    mode='a',
    rotation='5 MB'
)

def train_and_evaluate_model(X, y, test_size, eta0, random_seed_array):
    """
    Train the model and evaluate performance.

    Parameters:
    X (pd.DataFrame): Features.
    y (pd.Series): Target variable.
    test_size (float): Proportion of the test set.
    eta0 (float): Learning rate.
    random_seed_array (iterable): List of random seeds.

    Returns:
    list: List of accuracies.
    list: List of precisions.
    """
    accuracy_array = []
    precision_array = []

    logger.info(f"Starting training and evaluation with test_size={test_size}, eta0={eta0}")

    for random_state in random_seed_array:
        logger.debug(f"Training with random_state={random_state}")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state
            )
            # Data scaling
            scaler = StandardScaler()
            X_train_std = scaler.fit_transform(X_train)
            X_test_std = scaler.transform(X_test)

            model = Perceptron(eta0=eta0)
            model.fit(X_train_std, y_train)
            y_pred = model.predict(X_test_std)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)

            accuracy_array.append(accuracy)
            precision_array.append(precision)

            logger.debug(f"Completed training with random_state={random_state}. Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")

        except Exception as e:
            logger.exception(f"Error during training with random_state={random_state}: {e}")
            continue

    logger.debug("Completed training and evaluation.")
    return accuracy_array, precision_array

def train_final_model(X, y, eta0):
    """
    Train the final model on the entire dataset.

    Parameters:
    X (pd.DataFrame): Features.
    y (pd.Series): Target variable.
    eta0 (float): Learning rate.

    Returns:
    model: Trained model.
    scaler: Fitted scaler.
    """
    try:
        logger.debug(f"Starting training of final model with eta0={eta0}")
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        model = Perceptron(eta0=eta0)
        model.fit(X_std, y)

        logger.debug("Completed training of final model.")

        return model, scaler

    except Exception as e:
        logger.exception(f"Error during training of final model: {e}")
        raise
