from StratOptimizer import *
from strategies_to_optimize import *

df_path='/path/to/data'

df=pd.read_csv(df_path, parse_dates=['Date'], index_col='Date')
res_folder='/path/to/save/results'

params={'aroon_window':(10,50,5), 'bbands_window':(10,50,5), 'bbands_dev_up':(1,4,1), 'bbands_dev_down':(1,4,1),
                             'ema_window_long':(110,210,10), 'ema_window_short':(20,100,10)}
optimizer=StratOptimizer(df,15,15,0.7,1000,'EURUSD',aroon_bbands_ema,params,res_folder )
optimizer.optimize_strategy()
