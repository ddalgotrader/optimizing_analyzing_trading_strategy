import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neighbors import KDTree
import plotly.express as px


class StrategyAnalyzer():

    def __init__(self, results, metric):

        self.results = results
        self.results = self.results.reset_index()
        self.results = self.results.drop('index', axis=1)
        self.metric = metric

    def sort_agg_results(self, sorted_by='mean'):

        temp_df = self.results.copy()
        metric_cols_mean = temp_df.columns.str.contains(self.metric)

        temp_df[f'{self.metric}_mean'] = temp_df.loc[:, metric_cols_mean].mean(axis=1)

        metric_cols_std = temp_df.columns.str.contains(self.metric)
        temp_df[f'{self.metric}_std'] = temp_df.loc[:, metric_cols_std].std(axis=1)

        kurtosis_cols = temp_df.columns.str.contains('kurtosis')

        temp_df['kurtosis_mean'] = temp_df.loc[:, kurtosis_cols].mean(axis=1)
        skew_cols = temp_df.columns.str.contains('skew')
        temp_df['skew_mean'] = temp_df.loc[:, skew_cols].mean(axis=1)

        if self.metric in ['max_dd', 'stddev']:

            self.sorted_results = temp_df.sort_values(by=f'{self.metric}_{sorted_by}')

        else:

            if sorted_by == 'std':
                self.sorted_results = temp_df.sort_values(by=f'{self.metric}_{sorted_by}')
            else:

                self.sorted_results = temp_df.sort_values(by=f'{self.metric}_{sorted_by}', ascending=False)

    def describe_results(self):

        return self.sorted_results[
            [f'{self.metric}_mean', f'{self.metric}_std', 'kurtosis_mean', 'skew_mean']].describe()

    def plot_best(self, metric_mean_limit, metric_std_limit):

        if self.metric in ['max_dd', 'stddev']:

            self.sorted_results['best'] = np.where(((self.sorted_results[
                                                         f'{self.metric}_mean'] <= metric_mean_limit) & (
                                                            self.sorted_results[
                                                                f'{self.metric}_std'] <= metric_std_limit)),
                                                   'in limit', 'out_of_limit')

        else:

            self.sorted_results['best'] = np.where(((self.sorted_results[
                                                         f'{self.metric}_mean'] >= metric_mean_limit) & (
                                                            self.sorted_results[
                                                                f'{self.metric}_std'] <= metric_std_limit)),
                                                   'in limit', 'out_of_limit')
        fig = px.scatter(self.sorted_results, x=f"{self.metric}_mean", y=f"{self.metric}_std", color='best',
                         text=self.sorted_results.index, size_max=60)

        fig.update_traces(textposition='top center')

        if self.metric in ['max_dd', 'stddev']:

            fig.add_shape(type="line",
                          x0=metric_mean_limit, y0=0, x1=metric_mean_limit, y1=metric_std_limit,
                          line=dict(color="yellow", width=3, dash="dashdot"))

            fig.add_shape(type="line",
                          x0=0, y0=metric_std_limit, x1=metric_mean_limit, y1=metric_std_limit,
                          line=dict(color="yellow", width=3, dash="dashdot"))
            
            

        else:
            fig.add_shape(type="line",
                          x0=metric_mean_limit, y0=0, x1=metric_mean_limit, y1=metric_std_limit,
                          line=dict(color="yellow", width=3, dash="dashdot"))

            fig.add_shape(type="line",
                          x0=metric_mean_limit, y0=metric_std_limit, x1=fig.data[0].x.max(), y1=metric_std_limit,
                          line=dict(color="yellow", width=3, dash="dashdot"))

        fig.update_layout(
            height=800,
            title_text=f'{self.metric}_mean vs {self.metric}_std'
        )

        fig.show()
        

    def find_k_near_best(self, ref_mean, ref_std, n_nearest):

        self.ref_point = [ref_mean, ref_std]
        X = np.array(self.sorted_results[[f'{self.metric}_mean', f'{self.metric}_std']])
        tree = KDTree(X, leaf_size=2)
        dist, ind = tree.query([self.ref_point], k=n_nearest)

        self.k_near_best = self.sorted_results.iloc[ind[0]]

    def plot_k_near_best(self):

        fig = px.scatter(self.k_near_best, x=f"{self.metric}_mean", y=f"{self.metric}_std", color='best', log_x=True,
                         size_max=60)

        fig.update_traces(textposition='top center')

        fig.add_trace(go.Scatter(x=[self.ref_point[0]], y=[self.ref_point[1]], marker_size=15, marker_color='green',
                                 name='reference_point'))

        fig.update_layout(
            height=800,
            title_text=f'{self.metric}_mean vs {self.metric}_std'
        )

        fig.show()

    def find_best_normal_dist(self):

        X = self.k_near_best[['kurtosis_mean', 'skew_mean']]

        tree = KDTree(X, leaf_size=2)

        optimal_kurtosis = 3
        optimal_skew = 0

        dist, ind = tree.query([[optimal_kurtosis, optimal_skew]], k=len(self.k_near_best))

        self.normal_best = self.k_near_best.iloc[ind[0]]

    def plot_params(self, param):

        fig = px.scatter(self.sorted_results, x=f"{self.metric}_mean", y=f"{self.metric}_std", color=param,
                         text=self.sorted_results.index, log_x=True, size_max=60)

        fig.update_traces(textposition='top center')

        fig.update_layout(
            height=800,
            title_text=f'{self.metric}_mean vs {self.metric}_std'
        )

        fig.show()
        

