
import pandas as pd
import numpy as np
import csv
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import pprint
from sklearn.preprocessing import StandardScaler


def mse(y, yhat):
    '''
    compute mse from actual and forecasted/fitted data
    input:
        y           actual data
        yhat        forecasted/fitted

    return:
        mse
    '''
    if len(y) != len(yhat):
        print("error in MSE; len y != len yhat")
        return None

    df = pd.DataFrame(data=zip(y, yhat))
    df2 = df.dropna()
    df["diff"] = df[0] - df[1]

    if (len(df2) < len(df) / 2):  # skip if too many Nones in fitted model
        mse = None
    else:
        mse = (df["diff"] ** 2).sum() / len(df)

    return mse


def read_excel(filename):
    '''
    input: excel filename  (data for many players)
    output:
        df       pandas dataframe
        players  set of player names for iterating over players
    '''

    df = pd.read_excel(filename)
    players = set(df["PLAYER_NAME"])

    return df, players


def fit_markov(data, n_regimes=1, n_order=1, print_results=False):
    '''
    fit markov regime-switching model to time series
    inputs:
        data        time series to fit
        n_regimes   number of markov states
        n_order     order of AR

    returns:
        fitted      array of model estimates
                        (use collect_params to parse)
    '''
    mod = sm.tsa.MarkovAutoregression(endog=data, k_regimes=n_regimes, order=n_order, switching_ar=False)

    # can use mod.param_names to print parameter names
    try:
        fitted = mod.fit(return_params=False, maxiter=1000, search_reps=20, disp=False)
    except:
        fitted = None
        print("failure to fit---------------------------------")
    if (print_results):
        display(fitted.summary())

    return fitted


def fit_markov_exog(endog, exog, n_regimes=1, n_order=1, print_results=False):
    '''
    fit markov regime-switching model to time series
    inputs:
        data        time series to fit
        n_regimes   number of markov states
        n_order     order of AR

    returns:
        fitted      array of model estimates
                        (use collect_params to parse)
    '''
    mod = sm.tsa.MarkovAutoregression(endog=endog, exog=exog, k_regimes=n_regimes, order=n_order, switching_ar=False)

    try:
        fitted = mod.fit(return_params=False, maxiter=1000, method='basinhopping', disp=False)

    except:
        print("failure to fit exog---------------------------------")
        return None

    if (print_results):
        display(fitted.summary())

    return fitted


def collect_params(fitted):
    '''
    parse array of fitted parameters
    inputs:
        fitted      array of model estimates (from fit_markov)

    returns:
        p           matrix of transition probabiliites
        mu          vector of state-dependent means
        sigma       standard error
        phi         autoregression coefficient(s)
        n_regimes   number of markov states
        n_order     order of AR
    '''
    n_regimes = fitted.k_regimes
    n_order = fitted.order

    p = np.zeros((n_regimes, n_regimes))  # transition matrix
    mu = np.zeros((n_regimes))  # state dependent means
    phi = np.zeros((n_order))  # autoregressive coefficients

    t = 0  # index of input array

    ### extract p

    for i in range(n_regimes - 1):
        for j in range(n_regimes):  # copy params into p
            p[i][j] = fitted.params[t]
            t += 1

    s = np.sum(p, axis=0)  # to calculate residual probabilities by subtracting from 1

    for j in range(n_regimes):  # remaining transition probs
        p[n_regimes - 1][j] = 1 - s[j]

        # check that it adds up

    if not (np.sum(p, axis=1).all() == 1.0):
        print("WARNING sum of probs equals ", np.sum(p, axis=1).all())

    # extract mu

    for i in range(n_regimes):
        mu[i] = fitted.params[t]
        t += 1

    # extract sigma

    sigma = fitted.params[t]
    t += 1

    # extract phi

    for i in range(n_order):
        phi[i] = fitted.params[t]
        t += 1

    return p, mu, sigma, phi, n_regimes, n_order


def collect_params_exog(fitted):
    '''
    parse array of fitted parameters
    inputs:
        fitted      array of model estimates (from fit_markov)

    returns:
        p           matrix of transition probabiliites
        mu          vector of state-dependent means
        beta        coefficients on exogenous variables
        sigma       standard error
        phi         autoregression coefficient(s)
        n_regimes   number of markov states
        n_order     order of AR
    '''

    n_regimes = fitted.k_regimes
    n_order = fitted.order
    n_exog = len(fitted.params) - ((n_regimes) * (n_regimes - 1) + n_regimes + 1 + n_order)

    p = np.zeros((n_regimes, n_regimes))  # transition matrix
    mu = np.zeros((n_regimes))  # state dependent means
    beta = np.zeros((n_exog))  # coefficients on exogenous variables
    phi = np.zeros((n_order))  # autoregressive coefficients

    t = 0  # index of input array

    ### extract p

    for i in range(n_regimes - 1):
        for j in range(n_regimes):  # copy params into p
            p[i][j] = fitted.params[t]
            t += 1

    s = np.sum(p, axis=0)  # to calculate residual probabilities by subtracting from 1

    for j in range(n_regimes):  # remaining transition probs
        p[n_regimes - 1][j] = 1 - s[j]

        # check that it adds up

    if not (np.sum(p, axis=1).all() == 1.0):
        print("WARNING sum of probs equals ", np.sum(p, axis=1).all())

    # extract mu

    for i in range(n_regimes):
        mu[i] = fitted.params[t]
        t += 1

    # extract beta

    for i in range(n_exog):
        beta[i] = fitted.params[t]
        t += 1

    # extract sigma

    sigma = fitted.params[t]
    t += 1

    # extract phi

    for i in range(n_order):
        phi[i] = fitted.params[t]
        t += 1

    return p, mu, beta, sigma, phi, n_regimes, n_order


def plot_filt_p(fitted, player):
    '''
    plot historical filtered fitted probabilities from "fit" object
    inputs:
        fitted      array of model estimates (from fit_markov)
        player      player ID or name

    returns: None (generates plot inside)

    '''

    x = [i for i in range(len(fitted.filtered_marginal_probabilities))]

    _, mu, _, _, n_regimes, _ = collect_params(fitted)

    fig, ax = plt.subplots()

    for i in range(n_regimes):
        p0 = [fitted.filtered_marginal_probabilities[j][i] for j in range(len(fitted.filtered_marginal_probabilities))]
        ax.plot(x, p0, label="P[" + str(i) + "], mu = " + str(round(mu[i], 2)))

    plt.legend()
    ax.set_xlabel('game')
    ax.set_ylabel('state probability')
    ax.set_title("Filtered Probabilities for Player " + str(player))

    plt.show()

    return None


def plot_smooth_p(fitted, player):
    '''
    plot historical smoothed fitted probabilities from "fit" object
    inputs:
        fitted      array of model estimates (from fit_markov)
        player      player ID or name

    returns: None (generates plot inside)

    '''
    x = [i for i in range(len(fitted.smoothed_marginal_probabilities))]

    _, mu, _, _, n_regimes, _ = collect_params(fitted)

    fig, ax = plt.subplots()

    for i in range(n_regimes):
        p0 = [fitted.smoothed_marginal_probabilities[j][i] for j in range(len(fitted.smoothed_marginal_probabilities))]
        ax.plot(x, p0, label="P[" + str(i) + "], mu = " + str(round(mu[i], 2)))

    ax.set_xlabel('game')
    ax.set_ylabel('state probability');
    ax.set_title("Smoothed Probabilities for Player " + str(player))

    plt.legend()
    plt.show()

    return None


def onestep_forecast_markov(fitted, yt=None):
    '''
    do one step ahead forecast for markov model
        y(t+1) = E(mu(t+1)) + phi*y(t)
        where E(mu(t+1)) = sum_i p(i,t+1)*mu(i)
        and p(i,t+1) = filtered probability of state i for date t+1

    inputs:
        fitted      array of model estimates (from fit_markov)
        yt          current observation (for AR component)

    output:
        one step ahead forecast as above


    '''
    p, mu, _, phi, n_regimes, n_order = collect_params(fitted)

    T = len(fitted.filtered_marginal_probabilities) - 1
    pfilt = [fitted.filtered_marginal_probabilities[T][i] for i in range(n_regimes)]

    p0 = np.matmul(p, pfilt)

    Emu = 0

    for i in range(n_regimes):
        Emu += mu[i] * p0[i]

    return Emu + phi * yt


def nstep_forecast_markov(fitted, yt=None, n=1):
    '''
    do n step ahead forecast:
        y(t+n) = E(mu(t+n)) + phi*y(t)
        where E(mu(t+n)) = sum_i {p(i,t+n)*mu(i)}
        and p(i,t+n) = filtered probability of state i for date t+n

    inputs:
        fitted      array of model estimates (from fit_markov)
        yt          current observation (for AR component)

    output:
        n step ahead forecast as above

    reference: Hamilton, Time Series Analysis, p. 695 et seq

    '''
    p, mu, _, phi, n_regimes, n_order = collect_params(fitted)

    T = len(fitted.filtered_marginal_probabilities) - 1
    pfilt = [fitted.filtered_marginal_probabilities[T][i] for i in range(n_regimes)]

    pn = np.linalg.matrix_power(p, n)

    p0 = np.matmul(pn, pfilt)

    Emu = 0

    for i in range(n_regimes):
        Emu += mu[i] * p0[i]

    return Emu + phi * yt


def nstep_forecast_markov_exog(fitted, yt=None, xt=None, n=1):
    '''
    do n step ahead forecast:
        y(t+n) = E(mu(t+n)) + phi*y(t) + sum (beta(i) x(i,t))
        where E(mu(t+n)) = sum_i {p(i,t+n)*mu(i)}
        x and beta are exogenous variables and their coefficients
        and p(i,t+n) = filtered probability of state i for date t+n

    inputs:
        fitted      array of model estimates (from fit_markov)
        yt          current observation (for AR component)
        xt.         current value of exog variables (not forecasted)

    output:
        n step ahead forecast as above

    reference: Hamilton, Time Series Analysis, p. 695 et seq

    '''
    p, mu, beta, _, phi, n_regimes, n_order = collect_params_exog(fitted)

    T = len(fitted.filtered_marginal_probabilities) - 1
    pfilt = [fitted.filtered_marginal_probabilities[T][i] for i in range(n_regimes)]

    pn = np.linalg.matrix_power(p, n)

    p0 = np.matmul(pn, pfilt)

    Emu = 0

    for i in range(n_regimes):
        Emu += mu[i] * p0[i]

    bx = 0

    for i in range(len(xt)):
        bx += beta[i] * xt[i]

    return Emu + phi * yt + bx


def model_select_bic(data):
    '''
    select model parameterization based on BIC
    compares BIC for AR(1), 2-state Markov AR(1), 3-state Markov AR(1)
    skips model if std errors are nan (=> identification problem)

    input:
        data        time series to fit
    output:
        n_star      BIC-optimal number of regimes
        bic_star    optimal BIC

    '''

    # fit AR ("1-regime") model

    bic_cutoff = 10  # cutoff for delta in bic

    n_star = 1

    mod = AutoReg(data, 1, old_names=False)
    fitted = mod.fit()
    bic_AR = fitted.bic
    bic_star = bic_AR

    # fit two regime model

    n_regimes = 2

    fitted = fit_markov(data, n_regimes, n_order=1, print_results=False)

    if (fitted is not None):
        bic_2state = fitted.bic

        if (not (np.isnan(fitted.conf_int()).any())):
            if abs(bic_AR - bic_2state) > bic_cutoff:  # could use 20 cutoff
                n_star = n_regimes
                bic_star = bic_2state

    # fit 3 regime model

    n_regimes = 3

    fitted = fit_markov(data, n_regimes, n_order=1, print_results=False)

    if (fitted is not None):
        bic_3state = fitted.bic

        if (not (np.isnan(fitted.conf_int()).any())):
            if ((abs(bic_2state - bic_3state) > bic_cutoff) and (
                    abs(bic_AR - bic_3state) > bic_cutoff)):  # could use 20 cutoff
                n_star = n_regimes
                bic_star = bic_3state

    return n_star, bic_star


def model_select_mse_onestep(data):
    '''
    select model parameterization based on one step ahead mse
    compares mse for AR(1), 2-state Markov AR(1), 3-state Markov AR(1)
    skips model if std errors are nan (=> identification problem)

    input:
        data        time series to fit
    output:
        n_regimes   mse-optimal number of regimes

    '''

    # generate sequence of datasets for (0, T-n_forecasts)...(n_forecasts,T)
    # where n_forecasts is the number of forecasts going into the mse

    n_forecasts = 10

    y_for_mse = []
    yhat_for_mse_AR = []  # to store actual and forecast
    yhat_for_mse_markov_2state = []
    yhat_for_mse_markov_3state = []

    n_star = 1

    for i in range(len(data) - n_forecasts, len(data)):
        # break out training data
        T_end = i
        T_start = i - (len(data) - n_forecasts)

        data_training = data[T_start:T_end]

        y_T1 = data[T_end]  # y(t+1) used to compute MSE
        y_for_mse.append(y_T1)

        y_T = data_training[len(data_training) - 1]  # y(t), used for forecasting

        # run AR model
        mod = AutoReg(data_training, 1, old_names=False)
        fitted = mod.fit()
        yhat = onestep_forecast_AR(fitted, y_T)
        yhat_for_mse_AR.append(yhat)

        # run Markov 2 state

        n_regimes = 2

        try:
            fitted = fit_markov(data, n_regimes, n_order=1, print_results=False)
        except:
            fitted = None
            print(' failure to fit in markov n = 2 -----------')

        if (fitted is not None):
            yhat = onestep_forecast_markov(fitted, y_T)
            if (not (np.isnan(fitted.conf_int()).any())):
                yhat_for_mse_markov_2state.append(yhat)
            else:
                yhat_for_mse_markov_2state.append(None)

        # run Markov 3 state

        n_regimes = 3

        try:
            fitted = fit_markov(data, n_regimes, n_order=1, print_results=False)
        except:
            fitted = None
            print('failed to fit in markov n = 3 ---------------')

        if (fitted is not None):
            yhat = onestep_forecast_markov(fitted, y_T)

            if (not (np.isnan(fitted.conf_int()).any())):
                yhat_for_mse_markov_3state.append(yhat)
            else:
                yhat_for_mse_markov_3state.append(None)

    mse_AR = mse(y_for_mse, yhat_for_mse_AR)
    mse_m2 = mse(y_for_mse, yhat_for_mse_markov_2state)
    mse_m3 = mse(y_for_mse, yhat_for_mse_markov_3state)

    mse_models = [m for m in [mse_AR, mse_m2, mse_m3] if m is not None]

    if (mse_models):  # if it's not null
        if min(mse_models) == mse_AR:
            n_star = 1
            mse_star = mse_AR
        elif min(mse_models) == mse_m2:
            n_star = 2
            mse_star = mse_m2
        else:
            n_star = 3
            mse_star = mse_m3

    return n_star, mse_star


def model_select_mse_onestep_exog(endog, exog):
    '''
    select model parameterization based on one step ahead mse
    compares mse for AR(1), 2-state Markov AR(1), 3-state Markov AR(1)
    now with exogenous variables
    skips model if std errors are nan (=> identification problem)

    input:
        endog,exog  time series to fit (y and X)
    output:
        n_regimes   mse-optimal number of regimes

    '''

    # generate sequence of datasets for (0, T-n_forecasts)...(n_forecasts,T)
    # where n_forecasts is the number of forecasts going into the mse

    n_forecasts = 10

    y_for_mse = []
    yhat_for_mse_AR = []  # to store actual and forecast
    yhat_for_mse_markov_2state = []
    yhat_for_mse_markov_3state = []

    n_star = 1

    for i in range(len(endog) - n_forecasts, len(endog)):
        # break out training data
        T_end = i
        T_start = i - (len(endog) - n_forecasts)

        endog_training = endog[T_start:T_end]
        exog_training = exog[T_start:T_end]

        y_T1 = endog[T_end]  # y(t+1) used to compute MSE
        y_for_mse.append(y_T1)

        y_T = endog_training[len(endog_training) - 1]  # y(t), used for forecasting
        X_T = exog_training[len(exog_training) - 1]  # x(t), used for forecasting

        # run AR model
        mod = AutoReg(endog_training, 1, old_names=False)
        fitted = mod.fit()
        yhat = onestep_forecast_AR(fitted, y_T)
        yhat_for_mse_AR.append(yhat)

        # run Markov 2 state

        n_regimes = 2

        try:
            fitted = fit_markov_exog(endog_training, exog_training, n_regimes, n_order=1, print_results=False)
        except:
            fitted = None
            print(' failure to fit in markov n = 2 -----------')

        if (fitted is not None):
            yhat = nstep_forecast_markov_exog(fitted, y_T, X_T, n=1)
            if (not (np.isnan(fitted.conf_int()).any())):
                yhat_for_mse_markov_2state.append(yhat)
            else:
                yhat_for_mse_markov_2state.append(None)

        # run Markov 3 state

        n_regimes = 3

        try:
            fitted = fit_markov_exog(endog_training, exog_training, n_regimes, n_order=1, print_results=False)
        except:
            fitted = None
            print('failed to fit in markov n = 3 ---------------')

        if (fitted is not None):
            yhat = nstep_forecast_markov_exog(fitted, y_T, X_T, n=1)

            if (not (np.isnan(fitted.conf_int()).any())):
                yhat_for_mse_markov_3state.append(yhat)
            else:
                yhat_for_mse_markov_3state.append(None)

    mse_AR = mse(y_for_mse, yhat_for_mse_AR)
    mse_m2 = mse(y_for_mse, yhat_for_mse_markov_2state)
    mse_m3 = mse(y_for_mse, yhat_for_mse_markov_3state)

    mse_models = [m for m in [mse_AR, mse_m2, mse_m3] if m is not None]

    if (mse_models):  # if it's not null
        if min(mse_models) == mse_AR:
            n_star = 1
            mse_star = mse_AR
        elif min(mse_models) == mse_m2:
            n_star = 2
            mse_star = mse_m2
        else:
            n_star = 3
            mse_star = mse_m3

    return n_star, mse_star


def onestep_forecast_AR(fitted, yt=None):
    '''
    do one step ahead forecast for AR model
        y(t+1) = mu + phi*y(t)

    inputs:
        fitted      array of model estimates (from statmodel AutoReg)
        yt          current observation (for AR component)

    output:
        one step ahead forecast as above

    '''

    mu = fitted.params[0]
    phi = fitted.params[1]

    return mu + phi * yt


def which_state(fitted):
    '''
    determine which state the player is in
    inputs:
        fitted      array of model estimates (from fit_markov)

    returns:
        state       ("hot, cold, neutral")

    '''

    x = [i for i in range(len(fitted.filtered_marginal_probabilities))]

    _, mu, _, _, n_regimes, _ = collect_params(fitted)

    p0 = [fitted.filtered_marginal_probabilities[len(fitted.filtered_marginal_probabilities) - 1][i] for i in
          range(n_regimes)]

    state = state_label(p0, mu)

    return state


def state_label(p, mu):
    '''
     determine the label for the current state (from which_state)
     inputs:
         p      array of state probability estimates
         mu     array of state-dependent means

     returns:
         state       ("hot, cold, neutral")

     '''
    if np.isnan(p).any():
        return 'None'

    p_max_index = p.index((max(p)))

    mu_p_max = mu[p_max_index]

    if mu_p_max == max(mu):
        label = "Hot"
    elif mu_p_max == min(mu):
        label = "Cold"
    else:
        label = "Neutral"

    return label


