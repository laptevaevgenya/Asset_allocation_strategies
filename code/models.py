"""
This file contains functions for different asset allocation models.
They return asset weights based on allocation algorithm.
"""
import numpy as np
import pandas as pd

# import methods for minimum-variance Markowitz portfolio
from pypfopt import HRPOpt
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

minvar_kwargs = {"returns_data": True, "log_returns": True}


def equal_weights(returns: pd.DataFrame) -> dict:
    """Returns equal weights for each asset in data given"""
    n_assets = len(returns.columns)
    return dict(zip(returns.columns,
                    np.array([1 / n_assets for _ in range(n_assets)])
                    ))


def risk_parity_weights(returns: pd.DataFrame) -> dict:
    """Calculates asset weights under risk-parity approach"""
    returns_std = returns.std().pow(-1).to_dict()
    std_sum = sum(returns_std.values())
    weights = dict(zip(returns_std.keys(),
                       np.array(
                           list(
                               returns_std.values()
                           )
                       ) / std_sum
                       ))
    return weights


def min_variance_weights(returns: pd.DataFrame, short: bool = False) -> dict:
    """
    Estimates weights for minimum variance portfolio with Markowitz model
    """
    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(returns, **minvar_kwargs)
    S = risk_models.sample_cov(returns, **minvar_kwargs)

    # Optimize for mimimum volatility level
    ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1) if short else (0, 1))
    weights = ef.min_volatility()
    weights = ef.clean_weights()
    return weights


def black_litterman_weights(prices: pd.DataFrame, short: bool = False) -> dict:
    """Estimates asset weights via Black-Litterman model"""
    pass


def hrp_weights(returns: pd.DataFrame, **kwargs) -> dict:
    """Estimates asset weights via hierarchical risk parity model"""
    hrp = HRPOpt(returns)
    hrp.optimize()
    return hrp.clean_weights()
