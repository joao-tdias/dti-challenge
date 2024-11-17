import pandas as pd
from loguru import logger
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from constants.constants import LOGS_PATH, LOGS_LEVEL

logger.add(LOGS_PATH,
           level=LOGS_LEVEL,
           mode='a',
           rotation='5 MB'
           )

def predict_new_data(model, scaler, data):
    """
    Predict the class for new data.

    Parameters:
    model: Trained model.
    scaler: Fitted scaler.
    data (list of lists): New data samples.

    Returns:
    np.ndarray: Predicted classes.
    """
    columns = [
        'sample_code_number', 'clump_thickness', 'uniformity_of_cell_size',
        'uniformity_of_cell_shape', 'marginal_adhesion',
        'single_epithelial_cell_size', 'bare_nuclei', 'bland_chromatin',
        'normal_nucleoli', 'mitoses'
    ]
    try:
        data_df = pd.DataFrame(data, columns=columns)
        data_std = scaler.transform(data_df)
        pred = model.predict(data_std)
        return pred
    except Exception as e:
        logger.error(f'An error ocurred during prediction. {e}')