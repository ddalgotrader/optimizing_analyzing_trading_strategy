import pandas as pd
import matplotlib.pyplot as plt

class SplitData():
    def __init__(self, data, folds_num, offset):
        self.data=data
        self.folds_num=folds_num
        self.offset=offset
        
    
    
    def split_data(self):
        
        folds=[]
        fold_size=int(len(self.data)/(((self.folds_num-1)*(self.offset))+1))
        

        for i in range(0,self.folds_num,1):

            end_idx=(1*fold_size)+(i*self.offset*fold_size)
            start_idx=end_idx-fold_size
            if i == self.folds_num-1:
                end_idx=len(self.data)-1

            fold=[int(start_idx),int(end_idx)]



            folds.append(fold)
        self.folds=folds

        return self.folds
    
    def plot_folds(self):
        empty_bar_down=[]
        fold_data=[]
        empty_bar_up=[]
        
        for i in range(len(self.folds)):
            empty_bar_down.append(self.folds[i][0])
            fold_data.append(self.folds[i][-1]-self.folds[i][0])
            empty_bar_up.append(len(self.data)-self.folds[i][-1])
        
        
        fold_plot_data=pd.DataFrame({'empty_bar_down':empty_bar_down,
                                 'train_data':fold_data,
                                'empty_bar_up':empty_bar_up
                                    }, index=range(1,self.folds_num+1))
        
        
        fold_plot_data.plot(kind='barh', stacked=True, color=['white', 'green','white'])
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [1,2]
        plt.legend([handles[idx] for idx in order[:1]],[labels[idx] for idx in order[:1]], loc='upper left')
        plt.show()

        