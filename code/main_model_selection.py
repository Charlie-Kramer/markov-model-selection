import numpy as np
from utilities import *

scaler = StandardScaler()

# set these parameters first
filename = "/Users/charleskramer/github/CODE/data/AllPlayersGames_short.xlsx"
min_datalength = 30  # minimum number of obs for estimation
variable = "FG_PCT"  # series for modeling
exog_variables = ["REST_DAYS", "HOME/AWAY"] # exogenous variables
subtract_ma = False  # whether to subtract MA
ma_order = 2  # order of MA variable

df, players = read_excel(filename)  # read the data into pandas

# truncating players for test run

players = list(players)[:5]

i = 0
n_players = len(players)

n_small = 0
n1 = 0
n2 = 0
n3 = 0

n_endog = 0
n_exog = 0

n_forecasts = 3  # number of out of sample forecast steps

with open('markov_summary_gh.txt', 'w') as f:
    f.write("Player_name,model,result\n")

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
        exog_data = scaler.fit_transform(exog_data) #standardize exogenous variables

        if (subtract_ma):
            exog_data = exog_data[1:, :]

        if len(NBA_data) > min_datalength: #if enough data
            n_states_endog, mse_endog = model_select_mse_onestep(NBA_data)
            n_states_exog, mse_exog = model_select_mse_onestep_exog(NBA_data, exog_data)
            if (mse_endog < mse_exog):
                n_states = n_states_endog
                n_endog += 1
                line = player + ", Endog," + str(n_states) + "\n"
                f.write(line)
            else:
                n_states = n_states_exog
                n_exog += 1
                line = player + ", Exog," + str(n_states) + "\n"
                f.write(line)

            if (n_states == 1):
                n1 += 1
            elif (n_states == 2):
                n2 += 1
            elif (n_states == 3):
                n3 += 1

        else: #if insufficient data
            n_small += 1
            print("skipping, too few observations")
            line = player + ", Not enough data, None \n"
            f.write(line)

print("*********  summary   ********************************")
print("too little data: " + str(n_small))
print("one regime:      " + str(n1))
print("two regimes:     " + str(n2))
print("three regimes:   " + str(n3))
print("endog, exog:   " + str(n_endog) + " " + str(n_exog))
print("*****************************************************")


