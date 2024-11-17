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

def load_data(file_path):
    """
    Load and preprocess the dataset.

    Parameters:
    file_path (str): Path to the data file.

    Returns:
    pd.DataFrame: DataFrame containing features.
    pd.Series: Series containing the target variable.
    """
    column_names = [
        "sample_code_number",
        "clump_thickness",
        "uniformity_of_cell_size",
        "uniformity_of_cell_shape",
        "marginal_adhesion",
        "single_epithelial_cell_size",
        "bare_nuclei",
        "bland_chromatin",
        "normal_nucleoli",
        "mitoses",
        "class"
    ]

    try:
        raw_df = pd.read_csv(file_path, names=column_names, na_values=["?"])
        raw_df.dropna(inplace=True)
        cleaned_df = raw_df.apply(pd.to_numeric, errors='raise')

        # Feature engineering: mapping the classes
        cleaned_df["class"] = cleaned_df["class"].map({4: 0, 2: 1})

        y = cleaned_df.pop('class')
        X = cleaned_df
        logger.success('Data loaded.')
        return X, y
    except Exception as e:
        logger.error(f"Error while loading data: {e}")
        raise