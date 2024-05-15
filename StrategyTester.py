import pandas as pd
import numpy as np
import inspect
from Metrics import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class StrategyTester():

    def __init__(self, symbol, data, strategy_func=None):

        # Define function arguments
        self.symbol = symbol
        self.strategy_func = strategy_func
        self.results = None
        self.df_to_plot = None
        self.data = data

        # Extract arguments from strategy
        self.strategy_name = strategy_func.__name__
        self.func_args = [x for x in inspect.getfullargspec(self.strategy_func)[0] if x not in ['data', 'plot_data']]

    def __repr__(self):
        return f"{self.strategy_name.upper} backtester(symbol = {self.symbol}, strategy={self.strategy_func})"

    def test_strategy(self, **kwargs):

        # Check if arguments of strategy are defined correctly
        check_attr_error = False
        for i in range(len(self.func_args)):
            if self.func_args[i] not in kwargs.keys():
                print(
                    f'Define  correct parameters {self.func_args[i]} for {self.strategy_func.__name__} strategy range before test strategy')
                check_attr_error = True

        if check_attr_error == False:

            attrs_to_test = []

            for attr in self.func_args:

                if attr == 'freq':
                    setattr(self, attr, f'{kwargs[attr]}min')
                else:
                    setattr(self, attr, kwargs[attr])

            self.results = self.strategy_func(self.data, **kwargs)

            self.df_to_plot = self.results.copy()

            self.run_test()

            data = self.results.copy()
            data["creturns"] = data["returns"].cumsum().apply(np.exp)
            data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)

            self.results = data

            self.performance()
            return self.perform

    def run_test(self):
        ''' Runs the strategy backtest.
        '''

        data = self.results.copy()
        data["strategy"] = data["position"].shift(1) * data["returns"]

        data["trades"] = data.position.diff().fillna(0).abs()

        data.strategy = data.strategy - data.trades * (data.spread / 2)

        self.results = data

    def performance(self):

        buy_and_hold_ret = self.results['returns'].dropna()
        strategy_ret = self.results['strategy'].dropna()

        func_names = ['simple_return', 'mean_return', 'stddev', 'sharpe_ratio', 'sortino_ratio', 'max_dd',
                      'cagr', 'calmar_ratio', 'skew', 'kurtosis', 'kelly', 'win_loss_ratio', 'wl_return_ratio',
                      'trades_count']

        results = []
        for func in func_names:
            performance_dict = {}
            f = globals()[func]
            buy_and_hold_result = 0
            strategy_result = 0
            if func in ['win_loss_ratio', 'wl_return_ratio', 'trades_count']:

                strategy_result = round(f(self.results), 3)
            else:
                buy_and_hold_result = round(f(buy_and_hold_ret), 5)
                strategy_result = round(f(strategy_ret), 5)

            performance_dict['buy_and_hold'] = buy_and_hold_result
            performance_dict[f'{self.strategy_name}_strategy'] = strategy_result
            results.append(performance_dict)

        self.perform = pd.DataFrame(results, index=func_names)

    def plot_results(self):
        ''' Plots the performance of the trading strategy and compares to "buy and hold".
        '''
        if self.results is None:
            print("Test strategy before plot results")

        else:
            df_plot = self.results.copy()
            title = f'{self.symbol}'
            for attr in self.func_args:
                title = title + f'| {attr} = {getattr(self, attr)}'
            title = title

            figure = make_subplots(rows=1, cols=1)

            figure.add_trace(go.Scatter(x=df_plot.index, y=df_plot['creturns'], mode='lines', name='buy and hold'),
                             col=1, row=1)
            figure.add_trace(go.Scatter(x=df_plot.index, y=df_plot['cstrategy'], mode='lines',
                                        name=f'{self.strategy_name} strategy'), col=1, row=1)

            figure.update_layout(title=title, xaxis_rangeslider_visible=False, yaxis_visible=False)
            figure.update_xaxes(rangebreaks=[dict(bounds=['sat', 'mon'])])
            figure.show()