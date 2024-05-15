from itertools import product
from apply import apply
from tqdm import tqdm
import warnings
from StrategyTester import *
from SplitData import *
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=(SettingWithCopyWarning))
warnings.simplefilter(action='ignore', category=FutureWarning)

class StratOptimizer():

    def __init__(self, data, freq, folds_num, folds_offset, chunk_size, instrument, strategy, params_range,
                 result_folder):
        '''

        :param data: data with quaotations of asset with columns Open,High,Low,Close,vol,spread,pips
        :param freq: interval of quotations ->int
        :param folds_num: on how many parts data should be splitted->int
        :param folds_offset: how data should overlap each other from 0 to 1 ->float
        :param chunk_size: size of dataframe with results that should be saved in separate csv file ->int
        :param instrument: instrument ->str
        :param strategy: strategy to iotimize
        :param params_range: dict of paramters to combine -> dict
        :param result_folder: folder to save results ->str
        '''
        self.data = data
        self.freq = freq
        self.folds_num = folds_num
        self.folds_offset = folds_offset
        self.chunk_size = chunk_size
        self.instrument = instrument
        self.strategy = strategy
        strategy_name = strategy.__name__
        self.params_range = params_range
        self.result_folder = result_folder
        self.combine_params()
        self.split_data()
        self.create_chunks()
        print('StrateOptimizer')
        print('================')
        print(f'strategy - {strategy_name}')
        print(f'instrument - {self.instrument}')
        print(f'data range - {self.data.index[0]}-{self.data.index[-1]}')
        print(f'folds number - {self.folds_num}')
        print(f'number of possible params comb - {len(self.combs)}')
        print(f'number of chunks - {len(self.chunk_list)}')

    def combine_params(self):

        attr_dict = {}

        for attr in self.params_range.keys():

            if type(self.params_range[attr]) == list:
                attr_dict[attr] = self.params_range[attr]


            else:
                attr_dict[attr] = list(range(*self.params_range[attr]))

        prod = [x for x in apply(product, attr_dict.values())]
        combs = [dict(zip(attr_dict.keys(), p)) for p in prod]
        self.combs = combs

    def split_data(self):

        self.splitter = SplitData(self.data, self.folds_num, self.folds_offset)
        self.folds = self.splitter.split_data()

    def create_chunks(self):

        chunk_list = []
        temp_list = []
        for i in range(len(self.combs)):

            temp_list.append(self.combs[i])

            if (i > 0) & (i % self.chunk_size == 0):
                chunk_list.append(temp_list)
                temp_list = []
            if i == (len(self.combs) - 1):
                chunk_list.append(temp_list)
                temp_list = []

        self.chunk_list = chunk_list

    def optimize_strategy(self):

        result_df = pd.DataFrame()
        chunk_counter = 0
        for chunk in tqdm(self.chunk_list):
            print(f'\nCalculating scores for chunk {chunk_counter}')

            all_df = pd.DataFrame()
            for c in chunk:
                fold_counter = 0
                temp_params_df = pd.DataFrame(c, index=[0])
                for f in self.folds:
                    df_test = self.data.iloc[f[0]:f[1]]
                    tester = StrategyTester('EURUSD', df_test, strategy_func=self.strategy)
                    tester.test_strategy(freq=15, **c)

                    res = tester.test_strategy(freq=15, **c).iloc[:, 1]
                    new_idxs = [f'{idx}_{fold_counter}' for idx in res.index]
                    res.index = new_idxs
                    res_df = pd.DataFrame(res)
                    res_df = res_df.T
                    res_df = res_df.reset_index()
                    temp_params_df = pd.concat([temp_params_df, res_df], axis=1)
                    temp_params_df = temp_params_df.drop('index', axis='columns')
                    fold_counter += 1
                all_df = pd.concat([all_df, temp_params_df])

            all_df.to_csv(f'{self.result_folder}/all_results_chunk_{chunk_counter}.csv', index=False)
            chunk_counter += 1

            result_df = pd.concat([result_df, all_df])

        self.result_df = result_df.reset_index().drop('index', axis=1)
