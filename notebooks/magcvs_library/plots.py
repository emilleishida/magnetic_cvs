import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import combinations
import requests
import io
from astropy.timeseries import LombScargleMultiband
from IPython.display import clear_output
from .utils import tqdm2
from .science import find_candidates


# Light function for heavy corner plots:
def corner_plot(df: pd.DataFrame,
                df2: pd.DataFrame = None,
                data_labels: list[str] | None = None,
                normalize_hist: bool = True,
                alpha: float = .1
                ) -> None:
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


def modified_corner_plot(dfx: pd.DataFrame,
                         dfy: pd.DataFrame,
                         df2x: pd.DataFrame,
                         df2y: pd.DataFrame,
                         data_labels: list[str] | None = None
                         ) -> None:
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


def accuracy_versus_n_neighbors(positive: pd.DataFrame,
                                negative: pd.DataFrame,
                                positive_sample_size: int = 5,
                                negative_sample_size: int = 2_000,
                                N: int = 100,
                                n_neighbors_list: list[int] = [1, 2, 3],
                                **kwargs
                                ) -> None:
    """
    Function to evaluate and plot the distribution of the accuracy (proportion of candidates that are of the positive class) as a function of the number of neighbors and given other parameters.

    Parameters
    ---
        positive: pd.DataFrame
            Feature data of positive objects.

        negative: pd.DataFrame
            Feature data of negative objects in the feature space.

        positive_sample_size: int, optional
            Number of positive objects to be found that will be sampled from positive and put in the feature space with the negatives. The remaining positive objects will be used as inputs to the algorithm. Defaults to 5.

        negative_sample_size: _type_, optional
            Number of objects to be sampled from negative. Defaults to 2_000.

        N: int, optional
            Number of iterations on which to compute the accuracy. At each iteration, a new random sample is taken from negative and positive. Defaults to 100.

        n_neighbors_list: list, optional
            List of number of nearest neighbors to be explored. Defaults to [1, 2, 3].

        kwargs:
            Additional keyword arguments to be passed to the find_candidates function. (score_threshold, max_candidates, feature_names, kept_columns)
    """

    # Store accuracies for boxplot:
    all_accuracies = []
    for n_neighbors in tqdm2(n_neighbors_list):
        current_accuracies = []
        for _ in range(N):
            Ids_to_be_found = np.random.choice(np.unique(positive['objectId']), size=positive_sample_size, replace=False)
            positive_to_be_found, positive_algo_input = positive[positive['objectId'].isin(Ids_to_be_found)].drop_duplicates(subset='objectId'), positive[~positive['objectId'].isin(Ids_to_be_found)]
            negative_sample = negative.sample(n=negative_sample_size)
            feature_space = pd.concat([positive_to_be_found, negative_sample]).reset_index(drop=True)
            candidates = find_candidates(
                positive=positive_algo_input,
                feature_space=feature_space,
                n_neighbors=n_neighbors,
                **kwargs
            )
            if len(candidates) != 0:
                current_accuracies.append(len(candidates[candidates['class'] == 'positive']) / len(candidates) * 100)
            else:
                continue
        all_accuracies.append(current_accuracies)

    # Flatten data and convert to categorical for seaborn boxplot:
    df_plot = pd.DataFrame({
        'accuracy': np.concatenate(all_accuracies),
        'n_neighbors': np.concatenate([np.repeat(n, repeats=len(acc)) for n, acc in zip(n_neighbors_list, all_accuracies)])
    })
    df_plot['n_neighbors'] = pd.Categorical(
        df_plot['n_neighbors'],
        categories=n_neighbors_list,
        ordered=True
    )

    if 'score_threshold' in kwargs:
        score_threshold = kwargs['score_threshold']
    else:
        score_threshold = find_candidates.__kwdefaults__['score_threshold']
    plt.title(f'{positive_sample_size} CVs to be found\nScore threshold = {score_threshold}')
    sns.boxplot(x='n_neighbors', y='accuracy', data=df_plot, width=0.4, showfliers=False, fill=False, medianprops={"color": "r", "linewidth": 2}, whis=(5, 95))

    # Add count annotations on top of each box:
    for idx, acc_list in enumerate(all_accuracies):
        count = len(acc_list)
        plt.text(
            x=idx,
            y=102,
            s=f'N={count}',
            ha='center',
            va='bottom',
            fontsize=9,
            color='gray'
        )

    plt.ylim(0, 110)
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy distribution (%)')
    plt.grid(axis='y')

    return


def explore_params(positive: pd.DataFrame,
                   negative: pd.DataFrame,
                   nb_positive_objects_to_be_found: list[int] = [1, 2, 5],
                   score_thresholds: list[int] = [5, 6, 7, 8, 9, 10],
                   N: int = 100
                   ) -> None:
    """
    Wrapper function of the accuracy_versus_n_neighbors function to explore the parameters of the algorithm.

    Parameters
    ---
        positive: pd.DataFrame
            Feature data of positive objects.

        negative: pd.DataFrame
            Feature data of negative objects.

        nb_positive_objects_to_be_found: list[int]
            List of number of objects to sample from positive and to be found by the algorithm. Defaults to [1, 2, 5].

        score_thresholds: list[int]
            List of score thresholds to be used in the algorithm. Defaults to [5, 6, 7, 8, 9, 10, 11, 12].
        
        N: int
            Number of iterations on which to compute the accuracy. At each iteration, a new random sample is taken from negative and positive. Defaults to 100.
    """

    X = len(nb_positive_objects_to_be_found)
    Y = len(score_thresholds)

    plt.subplots(Y, X, figsize=(X*3, Y*3.5), sharex=True, sharey=True)
    subplot_index = 1
    for score_threshold in score_thresholds:
        for n in nb_positive_objects_to_be_found:
            print(f'{subplot_index}/{X*Y}')
            plt.subplot(Y, X, subplot_index)
            subplot_index += 1
            accuracy_versus_n_neighbors(positive=positive,
                                        negative=negative,
                                        positive_sample_size=n,
                                        score_threshold=score_threshold,
                                        N=N,
                                        kept_columns=['objectId', 'class'])
            print('')
            clear_output()
    plt.tight_layout()
    
    return


def plot_periodogram(objectId: str) -> None:
    """
    Plot the lightcurve and its associated Lomb-Scargle periodogram of an object given its Id.

    Parameters
    ---
        objectId: str
            The Id of the object to plot.
    """

    # Get the light curve data of the specified object:
    obj = pd.read_json(io.BytesIO(requests.post("https://api.fink-portal.org/api/v1/objects",
                                                json={"objectId": objectId,
                                                      "columns": "i:objectId,i:jd,i:magpsf,i:sigmapsf,i:fid",
                                                      "output-format": "json"
                                                      }
                                                ).content
                                  )
                       ).sort_values(by='i:jd')

    # Extract time, magnitude, magnitude error and filter:
    t = obj['i:jd'].values
    m = obj['i:magpsf'].values
    m_err = obj['i:sigmapsf'].values
    fid = obj['i:fid'].values

    # Compute the Lomb-Scargle periodogram:
    frequency, power = LombScargleMultiband(t, m, fid, m_err).autopower(nyquist_factor=1)

    plt.figure(figsize=(12, 4))
    plt.suptitle(objectId)

    # Plot the periodogram:
    plt.subplot(1, 2, 1)
    plt.plot(frequency, power)
    plt.title('Lomb-Scargle Periodogram')
    plt.xlabel('Frequency (day$^{-1}$)')
    plt.ylabel('Lomb-Scargle Power')

    # Plot the lightcurve:
    plt.subplot(1, 2, 2)
    plt.errorbar(t[fid == 1], m[fid == 1], yerr=m_err[fid == 1], fmt='o', c='g', label='g-band')
    plt.errorbar(t[fid == 2], m[fid == 2], yerr=m_err[fid == 2], fmt='o', c='r', label='r-band')
    plt.legend(loc='upper right')
    plt.title('Light Curve')
    plt.xlabel('JD')
    plt.ylabel('Magnitude')
    plt.gca().invert_yaxis()

    plt.tight_layout()

    return