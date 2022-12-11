# driver code--given output of model selection produce forecasts and state estimates

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utilities import *

scaler = StandardScaler() #for scaling exogenous variables

# set these parameters first
filename = "/Users/charleskramer/github/CODE/data/AllPlayersGames_short.xlsx"
min_datalength = 30  # minimum number of obs for estimation
variable = "FG_PCT"  # series for modeling
exog_variables = ["REST_DAYS", "HOME/AWAY"] #exogenous variables
subtract_ma = False  # whether to subtract MA
ma_order = 2  # order of MA variable

df, players = read_excel(filename)  # read the data into pandas

i = 0
n_players = len(players)

n_history = 0  # set to -1 for all history

df_model = pd.read_csv('markov_summary_gh.txt')

with open('forecasts_only_gh.csv', 'w') as f:
    f.write("Player_name,forecast,state\n")

    for player in players:  # loop thru list of players and estimate
        i += 1
        print("*****************************************************")
        print("player " + player + " number ", i, "of ", n_players)
        print("*****************************************************")

        NBA_data = df[df["PLAYER_NAME"] == player][variable]

        if (subtract_ma):
            NBA_data = NBA_data - NBA_data.rolling(ma_order).mean()
            NBA_data = NBA_data.dropna()

        NBA_data = NBA_data.to_numpy()

        exog_data = df[df["PLAYER_NAME"] == player][exog_variables].to_numpy()
        exog_data = np.array(exog_data, dtype=float)
        exog_data = scaler.fit_transform(exog_data)

        if (subtract_ma):
            exog_data = exog_data[1:, :]

        if (player in df_model["Player_name"].tolist()): #check if player is in the output from model selection
            model = df_model[df_model["Player_name"] == player]["model"].tolist()[0].strip()
        else:
            model = None
            state = 'None'

        if len(NBA_data) > min_datalength and model != "Not enough data" and model != None: #if enough data and we have a selected model
            pname = df_model[df_model["Player_name"] == player]["Player_name"].tolist()[0]
            result = int(df_model[df_model["Player_name"] == player]["result"].tolist()[0])
            if n_history == -1:
                history = NBA_data
            else:
                history = NBA_data[len(NBA_data) - n_history:]
            if result > 1: #e.g. if it is not a pure AR model for this player
                if model == "Exog":
                    fitted = fit_markov_exog(NBA_data, exog_data, n_regimes=result, n_order=1, print_results=False)
                    forecast = nstep_forecast_markov_exog(fitted, yt=NBA_data[len(NBA_data) - 1],
                                                          xt=exog_data[len(NBA_data) - 1], n=1)
                elif model == "Endog":
                    fitted = fit_markov(NBA_data, n_regimes=result, n_order=1, print_results=False)
                    forecast = nstep_forecast_markov(fitted, yt=NBA_data[len(NBA_data) - 1], n=1)

                state = which_state(fitted)


            else: #else if it is a pure AR model
                state = 'None'
                mod = AutoReg(NBA_data, 1, old_names=False)
                fitted = mod.fit()
                forecast = np.atleast_1d(onestep_forecast_AR(fitted, NBA_data[len(NBA_data) - 1]))

            history_string = ""  #add the historical data
            for val in history:
                history_string += str(val) + ","
            line = player + "," + history_string + str(forecast[0]) + "," + state + "\n"

        else: #if insufficient data or no selected model for this player
            history_string = ""
            state = 'None'
            if n_history == -1:
                history = NBA_data
            else:
                history = NBA_data[len(NBA_data) - n_history:]
            for val in history:
                history_string += str(val) + ","

            none_string = "None"
            line = player + "," + history_string + none_string + "," + state + "\n"

        f.write(line)

