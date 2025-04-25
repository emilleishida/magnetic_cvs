import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import combinations
from magcvs_library.utils import tqdm2
from magcvs_library.science import find_candidates


# Light function for heavy corner plots:
def corner_plot(df: pd.DataFrame, df2: pd.DataFrame = None, data_labels: list[str] | None = None, normalize_hist: bool = True, alpha: float = .1) -> None:
    """
    Make a corner plot which dimension is the number of columns of the DataFrame.  
    This function aims to be faster than other high level cornerplot functions such as seaborn's pairplot or corner.py in order to have higher dimensionnal plots quicker.

    Parameters
    ---
        df: pd.DataFrame
            DataFrame containing the data to plot.

        df2: pd.DataFrame, optional
            Second DataFrame with same columns as df to be plotted with a different color. Defaults to None.

        data_labels: list[str], optional
            To be used when df2 is not None. Should contain the two labels for the data from df1 and df2. Defaults to None.

        normalize_hist: bool, optional
            If True, the histograms will be normalized. Defaults to True.

        alpha: float, optional
            Transparency for the scatter plots. Defaults to .1.
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
    """
    Modified version of the corner_plot function to plot the data from two different filters. Data from filter x is plotted on the x axis and data from filter y is plotted on the y axis.  
    dfx, dfy, df2x and df2y should have the same columns.

    Parameters
    ---
        dfx: pd.DataFrame
            The positive class data from the g filter.

        dfy: pd.DataFrame
            The positive class data from the r filter.

        df2x: pd.DataFrame
            The negative class data from the g filter.

        df2y: pd.DataFrame
            The negative class data from the r filter.

        data_labels: list[str] | None, optional
            The labels for the positive and negative class data. Defaults to None.
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


def accuracy_versus_n_neighbors(positive_train, positive_test, negative, negative_sample_size=2_000, N=100, candidate_threshold=1, max_candidates=None, n_neighbors_list=np.arange(1, 11)):
    """
    Function to evaluate and plot the mean accuracy (proportion of candidates that are of the positive class) as a function of the number of neighbors and given other parameters.

    Parameters
    ---
        positive_train: pd.DataFrame
            Feature data of positive objects to be found by the algorithm among the negative ones in the feature space.
            
        positive_test: pd.DataFrame
            Feature data of positive objects to find neighbors for (input to the algorithm).
            
        negative: pd.DataFrame
            Feature data of negative objects in the feature space.

        negative_sample_size: _type_, optional
            Number of objects to be sampled from negative. Defaults to 2_000.
            
        N: int, optional
            Number of iterations on which to compute the mean. At each iteration, a new random sample is taken from negative. Defaults to 100.
            
        candidate_threshold: int, optional
            See find_candidates' documentation. Defaults to 1.
            
        max_candidates: int, optional
            See find_candidates' documentation. Defaults to 10.
            
        n_neighbors_list: list, optional
            List of number of nearest neighbors to be explored. Defaults to np.arange(1, 11).
    """

    # Store accuracies for boxplot:
    all_accuracies = []
    for n_neighbors in tqdm2(n_neighbors_list): # add tqdm2 here for progress bar
        current_accuracies = []
        for _ in range(N):
            negative_sample = negative.sample(n=negative_sample_size)
            feature_space = pd.concat([positive_train, negative_sample]).reset_index(drop=True)
            candidates = find_candidates(
                positive=positive_test,
                feature_space=feature_space,
                n_neighbors=n_neighbors,
                candidate_threshold=candidate_threshold,
                max_candidates=max_candidates
            )
            if len(candidates) != 0:
                current_accuracies.append(len(candidates[candidates['class'] == 'positive']) / len(candidates) * 100)
            else:
                current_accuracies.append(100)
        all_accuracies.append(current_accuracies)

    # Flatten data for seaborn
    df_plot = pd.DataFrame({
        'accuracy': np.concatenate(all_accuracies),
        'n_neighbors': np.repeat(n_neighbors_list, repeats=len(all_accuracies[0]))
    })

    plt.title(f'Candidate threshold = {candidate_threshold}\nMax candidates = {max_candidates}')
    sns.boxplot(x='n_neighbors', y='accuracy', data=df_plot, width=0.4, showfliers=True, fill=False, medianprops={"color": "r", "linewidth": 2}, whis=(5, 95))
    
    #sns.stripplot(df_plot, x="n_neighbors", y="accuracy", size=2, color=".3")
    #sns.violinplot(data=df_plot, x="n_neighbors", y="accuracy", fill=False, inner='point')
    
    plt.axhline(y=50, color='r', linestyle='--', alpha=.5, label='50% accuracy')
    plt.legend(loc='upper right')
    #plt.ylim(0, 100)
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy distribution(%)')
    plt.grid(axis='y')

    return