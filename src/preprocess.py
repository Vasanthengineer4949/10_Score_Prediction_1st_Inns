from utils import Utils
import numpy as np
import pandas as pd

def preprocess(data_path):

    data = pd.read_csv(data_path)
    print("Data Read")

    preprocessor = Utils(data)
    print("Utils Object Created")

    col_drop_data = preprocessor.drop_columns(data)
    print("Columns dropped")

    miss_data = preprocessor.missing_values_handling(col_drop_data)
    print("Missing Values Handled")

    common_teams_data = preprocessor.common_teams(miss_data)
    print("Common teams kept")

    enc_data = preprocessor.categorical_values_handling(common_teams_data)
    print("Categorical feature handled")

    return enc_data

print("Started")
path = "artifacts\data.csv"
enc_data = preprocess(path)
print("Complete")
enc_data.to_csv("artifacts/preprocessed.csv", index=False)
