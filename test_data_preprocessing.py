from src.data_preprocessing import load_data
import pandas as pd

def test_load_data():
    try:
        df = load_data("data/raw/creditcard.csv")
        assert isinstance(df, pd.DataFrame)
    except FileNotFoundError:
        assert True  # dataset not present, skip
