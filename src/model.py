import pandas as pd
from utils import Utils

data = pd.read_csv("artifacts/preprocessed.csv")
data = data.drop(["venue", "bat_team", "bowl_team", "batsman", "bowler"], axis=1)

X = data.drop(["total"], axis=1)
y = data["total"]

util = Utils(data)

X_train, X_test, y_train, y_test = util.split_data(X, y)
print("Splitting completed")
regressor = util.model_train(X_train, y_train)
print("Model Training Completed")

mae, mse, r2 = util.model_metric(regressor, X_test, y_test)
print(mae, mse, r2)

util.save_model(regressor, "artifacts/rf1.pkl")


