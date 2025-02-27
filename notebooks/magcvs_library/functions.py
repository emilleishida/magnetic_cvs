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
def corner_plot(df: pd.DataFrame, df2: pd.DataFrame = None, data_labels: list[str] | None = None, normalize_hist: bool = True) -> None:
    """Make a corner plot which dimension is the number of columns of the DataFrame.  
    This function aims to be faster than other high level cornerplot functions such as seaborn's pairplot or corner.py in order to have higher dimensionnal plots quicker.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot
        df2 (pd.DataFrame, optional): Second DataFrame with same columns as df to be plotted with a different color. Defaults to None.
        data_labels (list[str], optional): To be used when df2 is not None. Should contain the two labels for the data from df1 and df2. Defaults to None.
        normalize_hist (bool, optional): If True, the histograms will be normalized. Defaults to True.
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
            plt.scatter(df[feature_pairs[pair][0]], df[feature_pairs[pair][1]], s=1, alpha=0.1)
            if df2 is not None:
                plt.scatter(df2[feature_pairs[pair][0]], df2[feature_pairs[pair][1]], s=1, alpha=0.1)
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