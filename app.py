from fastapi import FastAPI
import joblib
import uvicorn
import numpy as np

app = FastAPI(debug=True)

@app.get('/')
def home():
    return {'Project': 'Score Predictor'}

@app.get('/predict')
def predict(
    Runs: int, 
    Wickets_Fell: int, 
    Overs_Completed: float,
    Runs_in_last_5_overs: int,
    Wickets_in_last_5_overs: int,
    Batting_team: int,
    Bowling_team: int,
    Venue: int,
    Batsman: int,
    Bowler: int):
    pred_model = joblib.load('artifacts/rf1.pkl')
    prediction = pred_model.predict([[
        Runs, Wickets_Fell, Overs_Completed, Runs_in_last_5_overs,
        Wickets_in_last_5_overs, Batting_team, Bowling_team,
        Venue, Batsman, Bowler
    ]])
    
    return {f"The predicted score is {round(prediction[0])}"}


if __name__=="__main__":
    uvicorn.run(app)