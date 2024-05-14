import pandas as pd
import numpy as np
import talib as ta

def aroon_bbands_ecross_neut(data,aroon_window=40, bbands_window=25, bbands_dev_up=2, bbands_dev_down=3,
                             ema_window_long=140, ema_window_short=90):
    df = data.copy()


    df['AROONOSC'] = ta.AROONOSC(df['high'], df['low'], aroon_window)

    df['AROONOSC_shift'] = df['AROONOSC'].shift(1)

    conditions_aroon = [((df['AROONOSC'] > 0) & (df['AROONOSC_shift'] < 0)),
                        ((df['AROONOSC'] < 0) & (df['AROONOSC_shift'] > 0))]

    values_aroon = [1, -1]

    df['position_ch_aroon'] = np.select(conditions_aroon, values_aroon, 0)
    df['position_aroon'] = df['position_ch_aroon'].replace(0, method='ffill')

    df['upperband'], df['middleband'], df['lowerband'] = ta.BBANDS(df['close'], bbands_window, bbands_dev_up,
                                                                   bbands_dev_down)

    df['close_shift'] = df['close'].shift(1)

    conditions_bbands = [(df['close'] <= df['lowerband']) & (df['close_shift'] > df['lowerband']),
                         (df['close'] >= df['upperband']) & (df['close_shift'] < df['upperband'])]

    values_bbands = [1, -1]

    df['position_ch_bbands'] = np.select(conditions_bbands, values_bbands, 0)

    df['EMA_long'] = ta.EMA(df['close'], ema_window_long)

    df['EMA_short'] = ta.EMA(df['close'], ema_window_short)

    df['EMA_long_shift'] = df['EMA_long'].shift()

    df['EMA_short_shift'] = df['EMA_short'].shift()

    conditions_ema = [(df['EMA_short'] > df['EMA_long']) & (df['EMA_short_shift'] < df['EMA_long_shift']),
                      (df['EMA_short'] < df['EMA_long']) & (df['EMA_short_shift'] > df['EMA_long_shift'])]

    values_ema = [1, -1]

    df['position_ch_ema'] = np.select(conditions_ema, values_ema, 0)

    conditions_comb = [(df['position_aroon'] == 1) & (df['position_ch_bbands'] == 1),
                       (df['position_aroon'] == -1) & (df['position_ch_bbands'] == -1), df['position_ch_ema'] != 0]

    values_comb = [1, -1, 2]

    df['position_ch'] = np.select(conditions_comb, values_comb, 0)

    df['position'] = df['position_ch'].replace(0, method='ffill')
    df['position'] = df['position'].replace({2: 0})

    return df
