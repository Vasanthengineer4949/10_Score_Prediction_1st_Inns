# Preprocess the dataset
import numpy as np
import pandas as pd
import json
import joblib

class Utils:

    def __init__(self, data):
        self.data = data

    def drop_columns(self, data):
        data = data.drop(
            ["mid", "date", "striker", "non-striker"], axis=1)
        return data
    
    def missing_values_handling(self, data):
        data = data.dropna().reset_index()
        return data

    def common_teams(self, data):
        teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 
        'Rajasthan Royals', 'Mumbai Indians', 'Kings XI Punjab', 
        'Royal Challengers Bangalore', 'Delhi Daredevils', 
        'Sunrisers Hyderabad']
        data = data[(data['bat_team'].isin(teams)) & (data['bowl_team'].isin(teams))]
        return data

    def categorical_values_handling(self, data):
        enc_columns = ['bat_team', 'bowl_team', 'venue', 'batsman', 'bowler']
        from sklearn.preprocessing import LabelEncoder
        enc = LabelEncoder()
        for i in enc_columns:
            enc_dict = {}
            enc_column_name = i+"_enc"
            data[enc_column_name] = enc.fit_transform(data[i])
            enc_dict[enc_column_name] = [[j, v] for j,v in enumerate(list(enc.classes_))]
            out_file = open(f"artifacts/{i}.json", "w")
            json.dump(enc_dict, out_file)
            out_file.close()
        return data
    
    def split_data(self, X, y):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0, shuffle=True
        )
        return X_train, X_test, y_train, y_test

    def model_train(self, X, y):
        from sklearn.ensemble import RandomForestRegressor
        reg = RandomForestRegressor(random_state=0)
        reg.fit(X,y)
        return reg

    def model_metric(self, model, X_test, y_test):
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mae, mse, r2

    def save_model(self, model, file_path):
        joblib.dump(model, file_path)



        
