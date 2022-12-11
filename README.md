# markov-model-selection

## Overview

This code performs two tasks:

1. Given data on performance for a set of NBA basketball players, select the best Markov chain (aka Markov-switching) time series model for each player among five candidates: pure AR model (one state), AR with two states, AR with three states, AR with two states and exogenous variables, and AR with three states and exogenous variables.
(see code/main_model_selection.py). The code writes the results to the file "markov_summary_gh.txt".

2. Given the output of the above, estimate the model for each player, and write historical data, a forecast of performance, and the estimated current state to "forecasts_only_gh.csv". (see main_estimate_forecast.py)

It uses the statsmodels package for estimation, adding a number of utility functions to extract parameters, compute model selection criteria, and produce forecasts (see code/utilities.py)

## Background on Markov switching models

This code implements a simple version of the Markov switching model  

$$y_{t+1} = \alpha_{s,t}  + \sum_i \beta_i x_{i,t} + \phi y_{t-1} + \epsilon_t$$ 

where $y$ is the endogenous variable, and $x$ is an optional set of exogenous variables. The intercept $\alpha$ depends on the state $s$, $s = 1,...S$.
In the context of basketball performance, $y$ is a measure of performance (here, field goal percentage),
the exogenous variables also explain performance (here, a dummy variable for home/away and a variable for 
the number of rest days) states could correspond to 'hot streaks' or 'cold streaks'.

A good reference on Markov switching models is ["Regime Switching Models"](https://econweb.ucsd.edu/~jhamilto/palgrav1.pdf), Palgrave Dictionary of Economics, James Hamilton (2005)

## How to run the code

Start with main_model_selection.py--set the following parameters as desired (lines 7-12):

    min_datalength = 30  # minimum number of obs for estimation. 
    variable = "FG_PCT"  # series for modeling. 
    exog_variables = ["REST_DAYS", "HOME/AWAY"] # exogenous variables. 
    subtract_ma = False  # whether to subtract moving average from the independent variable. 
    ma_order = 2  # order of MA variable. 

Note that for this demo version, I have limited the number of players to 5:

    players = list(players)[:5]
    
Comment out or remove this line to estimate the model for all players. Note, however, that this can run a long time for large datasets (one run with over 500 players and 1.5 million records ran over 24 hours on a laptop).  

Next, run main_estimate_forecast.py--ensure that you have the above parameters set the same way (lines 10-16). The output file "forecasts_only_gh.csv" will contain the results for each player.
