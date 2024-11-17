from io import StringIO
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.data_functions import load_data

TEST_FILES = os.path.join(os.path.dirname(__file__), 'temp_files')

def test_load_data():
    data = """1000025,5,1,1,1,2,1,3,1,1,2
        1002945,5,4,4,5,7,10,3,2,1,2
        1015425,3,1,1,1,2,?,3,1,1,2
        1016277,6,8,8,1,3,4,3,7,1,2
        1017023,4,1,1,3,2,1,3,1,1,2
        1017122,8,10,10,8,7,10,9,7,1,4
        1018099,1,1,1,1,2,10,3,1,1,2
        1018561,2,1,2,1,2,1,3,1,1,2
        1033078,2,1,1,1,2,1,1,1,5,2
        1033078,4,2,1,1,2,1,2,1,1,2
    """
    data_file = StringIO(data)

    X, y = load_data(data_file)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == 9
    assert set(y.unique()) == {0, 1}
    counts = y.value_counts()
    assert counts[0] == 1
    assert counts[1] == 8

if __name__ == '__main__':
    test_load_data()
    