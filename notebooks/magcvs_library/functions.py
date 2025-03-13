from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

# Wrapper for tqdm with default arguments:
def tqdm2(iterable, **kwargs):
    default_kwargs = {'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt}'}
    default_kwargs.update(kwargs)
    return tqdm(iterable, **default_kwargs)


# Light function for heavy corner plots:
def corner_plot(df: pd.DataFrame, df2: pd.DataFrame = None, data_labels: list[str] | None = None, normalize_hist: bool = True, alpha: float = .1) -> None:
    """Make a corner plot which dimension is the number of columns of the DataFrame.  
    This function aims to be faster than other high level cornerplot functions such as seaborn's pairplot or corner.py in order to have higher dimensionnal plots quicker.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot
        df2 (pd.DataFrame, optional): Second DataFrame with same columns as df to be plotted with a different color. Defaults to None.
        data_labels (list[str], optional): To be used when df2 is not None. Should contain the two labels for the data from df1 and df2. Defaults to None.
        normalize_hist (bool, optional): If True, the histograms will be normalized. Defaults to True.
        alpha (float, optional): Transparency for the scatter plots. Defaults to .1.
    """

    # Handling user input errors:
    if df2 is not None:
        if df.columns.tolist() != df2.columns.tolist():
            raise ValueError('Columns of df1 and df2 are not the same')

    n = len(df.columns) # Dimension of the corner plot

    fig = plt.figure(figsize=(n*2, n*2))

    # Plotting histograms:
    index = 1
    for feature_name in df.columns:
        plt.subplot(n, n, index)
        plt.hist(df[feature_name], bins=np.linspace(min(df[feature_name]), max(df[feature_name]), 30), density=normalize_hist, alpha=0.5)
        if df2 is not None:
            plt.hist(df2[feature_name], bins=np.linspace(min(df2[feature_name]), max(df2[feature_name]), 30), density=normalize_hist, alpha=0.5)
        if index == n*n:
            plt.xlabel(feature_name)
        else:
            plt.xticks([])
        plt.yticks([])
        index += n + 1

    # Plotting scatter plots:
    feature_pairs = tuple(combinations(df.columns, 2))
    pair = 0
    for i in range(1, n):
        for j in range(i, n):
            plt.subplot(n, n, j*n + i)
            plt.scatter(df[feature_pairs[pair][0]], df[feature_pairs[pair][1]], s=1, alpha=alpha)
            if df2 is not None:
                plt.scatter(df2[feature_pairs[pair][0]], df2[feature_pairs[pair][1]], s=1, alpha=alpha)
            if i == 1:
                plt.ylabel(feature_pairs[pair][1])
            else:
                plt.yticks([])
            if j == n - 1:
                plt.xlabel(feature_pairs[pair][0])
            else:
                plt.xticks([])
            pair += 1

    if data_labels is not None:
        fig.legend(data_labels, loc='center right')
    fig.tight_layout()

    return


def modified_corner_plot(dfx: pd.DataFrame, dfy: pd.DataFrame, df2x: pd.DataFrame, df2y: pd.DataFrame, data_labels: list[str] | None = None) -> None:
    """Modified version of the corner_plot function to plot the data from two different filters. Data from filter x is plotted on the x axis and data from filter y is plotted on the y axis.  
    dfx, dfy, df2x and df2y should have the same columns.

    Args:
        dfx (pd.DataFrame): The positive class data from the g filter.
        dfy (pd.DataFrame): The positive class data from the r filter.
        df2x (pd.DataFrame): The negative class data from the g filter.
        df2y (pd.DataFrame): The negative class data from the r filter.
        data_labels (list[str] | None, optional): The labels for the positive and negative class data. Defaults to None.
    """

    n = len(dfx.columns) # Dimension of the corner plot

    fig = plt.figure(figsize=(n*2, n*2))
    
    # Plotting diagonal:
    index = 1
    for feature_name in dfx.columns:
        plt.subplot(n, n, index)
        plt.scatter(dfx[feature_name], dfy[feature_name], s=1, alpha=0.1)
        plt.scatter(df2x[feature_name], df2y[feature_name], s=1, alpha=0.1)
        if index == n*n:
            plt.xlabel(feature_name)
        else:
            plt.xticks([])
        if index != 1:
            plt.yticks([])
        else:
            plt.ylabel(feature_name)
        index += n + 1

    # Plotting scatter plots:
    feature_pairs = tuple(combinations(dfx.columns, 2))
    pair = 0
    for i in range(1, n):
        for j in range(i, n):
            plt.subplot(n, n, j*n + i)
            plt.scatter(dfx[feature_pairs[pair][0]], dfy[feature_pairs[pair][1]], s=1, alpha=0.1)
            plt.scatter(df2x[feature_pairs[pair][0]], df2y[feature_pairs[pair][1]], s=1, alpha=0.1)
            if i == 1:
                plt.ylabel(feature_pairs[pair][1])
            else:
                plt.yticks([])
            if j == n - 1:
                plt.xlabel(feature_pairs[pair][0])
            else:
                plt.xticks([])
            pair += 1

    if data_labels is not None:
        fig.legend(data_labels, loc='center right')
    fig.tight_layout()

    return