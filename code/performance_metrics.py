import numpy as np
import pandas as pd
from typing import Union
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def portfolio_returns(w: Union[dict, pd.Series, pd.DataFrame],
                      asset_returns: pd.DataFrame) -> pd.Series:
    """
    Returns portfolio returns for given asset weights
    and asset returns
    """
    # rearrange tickers to check that tickers in weights
    # and tickers in returns have the same order
    tickers = asset_returns.columns

    if isinstance(w, dict):
        w = pd.Series({w[t] for t in tickers})
        return w @ asset_returns.T
    elif isinstance(w, pd.Series):
        w = w.reindex(tickers)
        return w @ asset_returns.T
    elif isinstance(w, pd.DataFrame):
        w = w[tickers]
        # if data and weights have different length
        first_obs = asset_returns.index.get_loc(w.index[0])
        return (w * asset_returns.iloc[first_obs:, :]).sum(axis=1)


def sharpe(returns: pd.Series,
           rf: Union[float, pd.Series] = 0.02,
           periods_in_year: int = 252) -> float:
    """
    Annualized Sharpe ratio from daily data. For details see
    "The Statistics of Sharpe Ratios" by Andrew Lo (2002)

    Parameters:
    -----------
    returns:pandas Series
        of asset or strategy returns
    rf:float or pandas Series, default 0.02 (for American market).
        Risk-free rate
    periods_in_year:int, default 252.  How many periods are there
        in calendar year (to annualize Sharpe Ratio). For monthly
        data: 12
    """
    rf = np.log(1 + rf) / periods_in_year
    if isinstance(rf, float):
        return np.sqrt(periods_in_year) * (returns.mean() - rf) / returns.std()
    else:
        # check that dates for rf match dates for asset returns
        data = pd.concat([returns, rf],
                         axis=1,
                         join='inner')
        data.columns = ['returns', 'rf']
        excess_return = data['returns'] - data['rf']
        return np.sqrt(periods_in_year) * (excess_return.mean()) / excess_return.std()


def max_drawdown(returns: pd.Series) -> float:
    """Returns maximum drawdown for given asset or portfolio returns data"""
    cumulative = (returns + 1).cumprod()
    peaks = cumulative.cummax()
    drawdowns = cumulative / peaks - 1
    return drawdowns.min()


def omega(returns: pd.Series,
          thresh: float = 0.02,
          periods_in_year: int = 252,
          parametric: bool = False) -> float:
    """
    Returns annualized Omega ratio ("A universal Performance Measure" by Keating (2002))

    Parameters:
    -----------
    returns: pandas Series
       Asset or portfolio returns
    thresh:float, default 0.02.
        The threshold to calculate Omega ratio (returns above thresh we consider
        to be gains and returns below we consider to be losses)
    periods_in_year: int, default 252.
        Number of trading periods in calendar year to annualize Omega ratio
    parametric: bool, default=False. Whether to compute parametric Omega
        ratio (assumption for returns distribution is necessary) or
        just a raw estimate
    """
    thresh_daily = (1 + thresh) ** (1 / periods_in_year) - 1
    gains = (returns[returns > thresh_daily] - thresh_daily).sum()
    losses = (thresh_daily - returns[returns < thresh_daily]).sum()
    return gains / losses


def turnover(weights: pd.DataFrame) -> float:
    """Returns max turnover for given DataFrame of asset weights"""
    changes = np.abs(weights.diff().iloc[1:])
    # how many times the porftolio was rebalanced
    n_changes = changes[changes.sum(axis=1) != 0].shape[0]
    return round(changes.sum().sum() / n_changes, 4)


def cer(returns: pd.Series,
        rf: Union[pd.Series, float] = 0.02,
        periods_in_year: int = 252,
        theta: float = 2) -> float:
    """
    Returns annualized CER (certainty-equivalent return) for given asset returns
    Parameters:
    -----------
    returns: pd.Series
        Asset or strategy returns.
    rf: float or pandas Series, default 0.02
        risk-free return.
    periods_in_year: int, default 252.
        Number of trading periods in calendar year to annualize returns and std
    theta: float, default 2.
        Hyperparameter for risk aversion (also known as lambda)
    """
    rf = np.log(1 + rf) / periods_in_year
    if isinstance(rf, float):
        excess_return = returns - rf
    else:
        rf = np.log(1 + rf)
        # check that dates for rf match dates for asset returns
        data = pd.concat([returns, rf],
                         axis=1,
                         join='inner')
        data.columns = ['returns', 'rf']
        excess_return = data['returns'] - data['rf']

    annualized_return = excess_return.mean() * periods_in_year
    annualized_std = np.sqrt(periods_in_year) * excess_return.var()
    return annualized_return - theta / 2 * annualized_std


def factor_regression(returns: pd.Series,
                      returns_name: str,
                      factor_returns: Union[pd.Series, pd.DataFrame],
                      factors_to_drop: Union[str, list] = 'cma',
                      alpha_only: bool = True
                      ):
    """
    Returns OLS alpha and factor coefficients estimates for strategy returns
    given factor returns., R^2 and ratio of OLS RMSE to RMSE from constant model
    (regression of strategy returns on constant). The aim is to assess
    the performance of factor regression.

    Parameters:
    ----------
    returns: pd.Series
        Asset or strategy returns.
    returns_name: str.
        Name of asset or strategy returns.
    factor_returns: pd.Series or pd.DataFrame.
        Factor returns.
    factors_to_drop: str or list of strings, default 'cma'.
        Factor returns to drop from factor_returns. Default 'cma',
    because I suppose this return factor to be strange and useless.
    alpha_only: bool, default True.
        Whether to return only alpha coefficient and it's p-value.
    """
    if all([not isinstance(factor_returns, pd.DataFrame),
            not isinstance(factor_returns, pd.DataFrame)]):
        raise ValueError(
            f'Factor returns and returns data '
            f'should be Pandas DataFrame or Pandas Series'
        )
    elif all([not isinstance(factor_returns.index,
                             pd.DatetimeIndex),
              not isinstance(factor_returns.index,
                             pd.DatetimeIndex)]):
        raise ValueError(
            f'Factor returns index and returns index'
            f'data should be Pandas DatetimeIndex'
        )

    returns.name = returns_name
    factors_to_drop = [x for x in factors_to_drop if x in factor_returns.columns]
    factor_returns.drop(columns=factors_to_drop,
                        inplace=True)
    data = pd.merge(left=returns,
                    right=factor_returns,
                    left_index=True,
                    right_index=True,
                    how='inner')
    y = data[returns_name]
    X = data.drop(columns=returns_name)
    const = np.ones(shape=(y.shape[0], 1))

    # fit_intercept=True by default
    lm = LinearRegression()
    lm.fit(X, y)
    lm_const = LinearRegression()
    lm_const.fit(const, y)

    y_pred = lm.predict(X)
    y_const_pred = lm_const.predict(const)
    lm_rmse = mean_squared_error(y, y_pred, squared=False)
    lm_const_rmse = mean_squared_error(y, y_const_pred, squared=False)
    rmse_ratio = lm_rmse / lm_const_rmse
    r2 = r2_score(y, y_pred)

    # calculate p-values (not provided by sklearn)
    coefs = np.append(lm.intercept_, lm.coef_)
    newX = pd.DataFrame({"alpha": np.ones(len(X))}).join(
        pd.DataFrame(X.reset_index(drop=True))
    )

    mse = (sum((y - y_pred) ** 2)) / (newX.shape[0] - newX.shape[1])
    var_b = mse * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = coefs / sd_b
    df = newX.shape[0] - newX.columns.shape[0]
    p_values = [2 * (1 - stats.t.cdf(np.abs(i), df))
                for i in ts_b]

    coef_dict = dict(zip(newX.columns,
                         list(zip(coefs, p_values))
                         ))
    if alpha_only:
        return coef_dict['alpha']
    return coef_dict


def sortino(returns: pd.Series,
            rf: Union[int, float] = 0.02,
            periods_in_year: int = 252
            ):
    """
    Returns Annualized Sortino Ratio for given portfolio returns (standard deviation of
    returns that are below given cutoff).

    Parameters:
    -----------
    returns: pandas Series.
        Pandas series of asset or portfolio returns.
    rf: int or float.
        Minimum acceptable return.
    periods_in_year: int.
        Number of periods in year to annualize Sortino Ratio.
    """
    rf = np.log(1 + rf) / periods_in_year
    if isinstance(rf, float):
        return np.sqrt(periods_in_year) * (returns.mean() - rf) / (returns[returns < rf]).std()
    else:
        if isinstance(rf, pd.DataFrame):
            rf = rf.squeeze()
        # check that dates for rf match dates for asset returns
        data = pd.concat([returns, rf],
                         axis=1,
                         join='inner')
        data.columns = ['returns', 'rf']
        excess_return = data['returns'] - data['rf']
        downside_risk = excess_return[excess_return.lt(rf)].std()
        return np.sqrt(periods_in_year) * (excess_return.mean()) / downside_risk


def calmar(returns: pd.Series,
           rf: Union[int, float] = 0.02,
           periods_in_year: int = 252
           ):
    """
    Returns annualized Calmar Ratio for given returns series.

    Parameters:
    -----------
    returns: pandas Series.
        Pandas series of asset or portfolio returns.
    rf: int or float.
        Minimum acceptable return.
    periods_in_year: int.
        Number of periods in year to annualize Sortino Ratio.
    """
    rf = np.log(1 + rf) / periods_in_year
    if isinstance(rf, pd.DataFrame):
        rf = rf.squeeze()
    excess_return = returns - rf
    annualized_return = ((1 + excess_return.mean()) ** periods_in_year) - 1
    return annualized_return / max_drawdown(returns) * -1
