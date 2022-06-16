"""
This script contains some additional simple models for asset allocations.
The models are built on the top of PyPortfolioopt Base
"""
import inspect
import pandas as pd
import numpy as np

import models
import performance_metrics as pf

from itertools import product
from typing import Union
from inspect import getfullargspec


# def cart_product(pools):
# """Alternative for itertools.product"""
# result = [[]]
# for pool in pools:
# result = [x + [y] for x in result for y in pool]
# return result


class BackTest:
    def __init__(self,
                 prices: pd.DataFrame = None,
                 log_returns: bool = True):
        """
        Class for backtesting of strategy
        Parameters:
        -----------
        prices: pandas DataFrame of asset prices. Column names are
            string tickers, index is pandas DateTimeIndex dates. No NaNs in data
            are allowed.
        log_returns:bool, default True. Whether to compute log returns or simple
            arithmetic returns.
        """
        self.prices = BackTest._validate_data(prices)

        if log_returns:
            returns = np.log(self.prices.pct_change().iloc[1:] + 1)
        else:
            returns = self.prices.pct_change().iloc[1:].dropna(inplace=True)

        self.returns = returns
        self.tickers = self.returns.columns
        self.strategy = None
        self.start = None
        self.step = None
        self.short = None
        self.weights_dict = dict.fromkeys(self.prices.index)
        self.weights_df = None
        self.strategy_returns = None
        self.perf_metrics = None

    @staticmethod
    def _validate_data(data):
        """
        Check returns data
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data is not pandas DataFrame")
        elif data.isna().sum().sum() != 0:
            raise ValueError("NaNs in data")
        elif any((type(x) != str for x in data.columns)):
            raise TypeError("Column names (tickers) are not of string datatype")
        elif not (isinstance(data.index, pd.DatetimeIndex)):
            raise TypeError("Index in data is not of pandas DateTimeIndex datatype")
        else:
            return data

    @staticmethod
    def _validate_strategy(strategy, short):
        """Checks if the strategy exists in models.py and shorts are possible if supplied"""
        strategy_func = strategy.replace('-', '_') + '_weights'
        if not hasattr(models, strategy_func):
            raise AttributeError("The strategy you supplied does not exist")
        elif short and "short" not in str(inspect.signature(getattr(models, strategy_func))):
            raise ValueError("Short positions are impossible for this strategy")

    def get_optimal_weights(self, subset):
        """
        Different asset allocation models in PyPortfolioopt have different methods
        for estimating optimal asset weights. Hence, we need some wrapping method
        to return optimal weights from strategy and data supplied by user.
        The method finds function for strategy in models.py by strategy name.
        """
        strategy_func = self.strategy.replace('-', '_') + '_weights'
        if self.short:
            return getattr(models, strategy_func)(subset, self.short)
        return getattr(models, strategy_func)(subset)

    def run(self,
            strategy: str,
            start: int,
            step: int,
            short: bool = False):
        """
        Backtest strategy on historical data with rolling window (step) from
        specific observation number (start)
        Parameters
        -----------
        strategy:str, type of asset allocation strategy. Possible values:
            * equal (1/N equally weighted portfolio)
            * risk-parity (simple risk parity)
            * min-vol (minimum variance portfolio for Markowitz model)
            * black-litterman (Black-Litterman allocation model)
            * hrp (hierarchical risk parity from PyPortfolioopt)
        start:int, from observation under which number to start backtesting.
        step:int, after each "step" number of observations the strategy weights are
            reestimated.
        """
        # check if strategy exists
        BackTest._validate_strategy(strategy, short)
        self.strategy, self.start, self.step, self.short = strategy, start, step, short

        i = self.start
        while i <= self.returns.shape[0] - 1:
            # !!! add subset with prices for black-litterman approach
            subset = self.returns.iloc[i - start:i]
            current_weights = self.get_optimal_weights(subset)

            # add weights to weight dict for estimation date
            date = self.returns.index[i]
            self.weights_dict[date] = current_weights
            i += step

        # convert dictionary with weights to dataframe
        self._weights_to_df()

    def _weights_to_df(self):
        if not self.weights_dict:
            raise ValueError("Dictionary with weights is empty, can't convert it to df")

        # Pyportfolioopt returns zeros for tickers with zero weights
        # so we can not care whether some of the tickers is missing in weights dict
        data_dict = {ticker: [x[ticker] if x else None \
                              for x in list(self.weights_dict.values())] \
                     for ticker in self.tickers}

        self.weights_df = pd.DataFrame.from_dict(data_dict)
        self.weights_df.index = list(self.weights_dict.keys())
        self.weights_df.index.name = 'date'
        self.weights_df.fillna(method='ffill', inplace=True)
        self.weights_df.dropna(inplace=True)

    def factor_regression(self,
                          factor_returns: Union[pd.Series, pd.DataFrame],
                          factors_to_drop: Union[str, list] = 'cma',
                          alpha_only: bool = True
                          ):
        """
        For documentation see factor_regression function in
        performance_metrics module.
        """
        return pf.factor_regression(pf.portfolio_returns(self.weights_df,
                                                         self.returns),
                                    self.strategy,
                                    factor_returns,
                                    factors_to_drop,
                                    alpha_only
                                    )

    def get_performance(self,
                        periods_in_year: int = 252,
                        rf: Union[pd.Series, float] = 0.02,
                        omega_thresh: float = 0.02,
                        theta: float = 2,
                        factor_returns: Union[pd.Series, pd.DataFrame] = None):
        """
        Returns performance metrics for strategy backtested with run().
        Parameters:
        -----------
        periods_in_year: int, default 252.
            Number of trading periods in year (default 252 for daily data)
        rf: pandas Series or float, default 0.02 (for American market)
            Risk-free rate.
        omega_thresh: float, default 0.02.
            Threshold return for Omega ratio. By default equals to US risk-free rate.
        theta: float, default 2.
            hyperparameter for CER ratio (also known as lambda)
        factor_returns: pd.Series or pd.DataFrame, default None.
            Factor returns to evaluate strategy performance and to estimate alpha.
        alpha_only: bool, True.
            Whether to return all the factor coefficients when running factor
            regression or just alpha and it's p-value.
        """
        self.strategy_returns = pf.portfolio_returns(self.weights_df, self.returns)

        if factor_returns is None:
            alpha, alpha_pvalue = None, None
        else:
            alpha, alpha_pvalue = pf.factor_regression(self.strategy_returns,
                                                       self.strategy,
                                                       factor_returns)

        self.perf_metrics = {'cumulative return': (self.strategy_returns + 1).cumprod().iloc[-1] - 1,
                             'annualized return': self.strategy_returns.mean() * periods_in_year,
                             'annualized vol': self.strategy_returns.std() * np.sqrt(periods_in_year),
                             'sharpe': pf.sharpe(self.strategy_returns,
                                                 rf,
                                                 periods_in_year),
                             'omega': pf.omega(self.strategy_returns,
                                               omega_thresh,
                                               periods_in_year),
                             'max drawdown': pf.max_drawdown(self.strategy_returns),
                             'turnover': 0 if self.strategy == 'equal' else pf.turnover(self.weights_df),
                             'cer': pf.cer(self.strategy_returns,
                                           rf,
                                           periods_in_year,
                                           theta),
                             'sortino': pf.sortino(self.strategy_returns,
                                                   rf,
                                                   periods_in_year),
                             'calmar': pf.calmar(self.strategy_returns,
                                                   rf,
                                                   periods_in_year),
                             'alpha': alpha,
                             'alpha_pvalue': alpha_pvalue}

        return self.perf_metrics


def _validate_parameters(func):
    """Parameter validation """

    def inner(self, *args):
        # get all Backtest.run() and Backtest.get_performance() arguments
        run_args = [x for x in getfullargspec(BackTest.run)[0] if x != 'self']
        perf_args = [x for x in getfullargspec(BackTest.get_performance)[0] if x != 'self']
        # args now is a tuple with 1 element
        strategies = list(set([x['strategy'] for x in args[0]]))
        if len(strategies) == 0:
            raise KeyError(
                "No argument 'strategy' was found inside parameters."
            )
        # get all arguments for each asset allocation strategy in parameters
        strategies_args = {}
        for s in strategies:
            strategy = s.replace('-', '_') + '_weights'
            strategies_args[strategy] = getfullargspec(getattr(models,
                                                               strategy)).args

        # check all parameters for each strategy
        # whether they are present in run_args, perf_args or strategies_args
        new_args = []
        for a in args[0]:
            s = strategies_args[a['strategy'].replace('-', '_') + '_weights']
            new_dict = {}
            for p, v in a.items():
                if p not in [*run_args, *perf_args, *s]:
                    print(
                        f"Parameter {p} not found in parameters of Backtest.run(), "
                        f"Backtest.get_performance() or model from models.py. "
                        f"Removing it."
                    )
                # some strategies can not short assets
                elif p == 'short' and 'short' not in s:
                    print(
                        f"For the strategy '{a['strategy']}' shorts are prohibited. "
                        f"Removing 'short' parameter from this strategy parameters."
                    )
                else:
                    new_dict[p] = v

            # save all correct parameters for each strategy in new list
            new_args = [*new_args, new_dict]
        # remove duplicate elements from list
        new_args = [i for n, i in enumerate(new_args)
                    if i not in new_args[(n + 1):]]
        return func(self, new_args)

    return inner


class GridBacktest:
    def __init__(self,
                 prices: Union[pd.Series, pd.DataFrame],
                 ):
        """
        Parameters:
        ----------
        prices: pd.DataFrame
            Prices of assets to build asset allocation models for.
        """
        self.prices = prices
        self.run_args = [x for x in getfullargspec(BackTest.run)[0] \
                         if x != 'self']
        self.perf_args = [x for x in getfullargspec(BackTest.get_performance)[0] \
                          if x != 'self']
        self.parameters = None
        self.strategies = None
        self.backtest_results = None

    def parameter_grid(self, param_dict):
        """Generates a list of dicts with parameter combinations"""
        k, v = param_dict.keys(), param_dict.values()
        v = [[x] if not isinstance(x, list) else x for x in v]
        return [dict(zip(k, val)) for val in product(*v)]

    @_validate_parameters
    def backtest(self,
                 parameters: list,
                 verbose=False):
        """
        Runs backtest for asset allocation strategies with asset prices
        and parameters given

        Parameters:
        -----------
        parameters: list.
            Python list of dicts with strategy parameters. Parameters come from:
                * Backtest.__init__() method
                * Backtest.run() method
                * Backtest.get_performance() method

            Example of parameters list:
            [{'strategy':'equal', 'start':252, 'step':21},
             {'strategy': 'hrp', 'start':252, 'step':63, 'rf':0.03}]

        verbose:bool, default False.
            Whether to print model names and parameters during backtesting.
        """
        self.parameters = parameters
        self.strategies = list(set([x['strategy'] for x in parameters]))
        backtest_results = []
        for i in self.parameters:
            if verbose:
                print(
                    f"Backtesting model '{i['strategy']} with parameters: "
                    f"{{k: v for k, v in i.items() if k in self.run_args}}"
                )
            Model = BackTest(self.prices)
            Model.run(**{k: v for k, v in i.items() if k in self.run_args})
            performance = Model.get_performance(**{k: v for k, v in i.items() if k in self.perf_args})
            backtest_results.append({**i, **performance})
        self.backtest_results = backtest_results

    def performance_table(self,
                          metric: str,
                          columns: Union[str, list] = None,
                          index: Union[str, list] = 'strategy',
                          ndigits: int = 3
                          ):
        """
        Returns dataframe with one performance metric calculated for
        all the strategies supplied. You can select, which strategy
        parameters to show in row indexes and which in columns.

        Parameters:
        -----------
        metric: str.
            Which performance metrics from Backtest.get_performance()
            to compare all the strategies by (for example, 'sharpe').
        columns: str or list of strins, default None.
            Which strategy characteristics to show in columns. If None,
            all the parameters for self.parameters that are present in
            Backtest.run() are used.
        index: str or list, default 'strategy'.
            Which strategy characteristics to show in rows.
        ngidits:int, default 3.
            How to round values in table.
        """
        # get all parameters from self.parameters which are in Backtest.run()
        keys = [metric,
                *[k for x in self.parameters
                  for k in x.keys()
                  if k in self.run_args]]
        keys = list(set(keys))
        data = [{k: x[k] for k in keys if k in x}
                for x in self.backtest_results]
        df = pd.DataFrame(data)
        # if column names are not given by the user
        if not columns:
            columns = [x for x in keys if 'strategy' not in x
                       and metric not in x]
        # fill empty cells for 'short' argument
        # if 'short' is prohibited for some strategies
        df.fillna(False, inplace=True)
        return pd.pivot_table(index=columns,
                              columns=index,
                              aggfunc=np.sum,
                              fill_value=0,
                              data=df).round(ndigits).replace(to_replace=0, value='--')
