import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from .managemcvs import get_mCVs_path


def fit_scale(positive: pd.DataFrame,
              unknown: pd.DataFrame,
              columns: list[str]
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize unknown using statistics from positive.

    Parameters
    ---
        positive: pd.DataFrame
            DataFrame containing features of the positive class objects.

        unknown: pd.DataFrame
            DataFrame containing features of the objects to evaluate.

        columns: list[str]
            List of feature names in the DataFrames columns to standardize.

    Returns
    ---
        positive_scaled: pd.DataFrame
            DataFrame with features of the positive class objects standardized.

        unknown_scaled: pd.DataFrame
            DataFrame with features of the objects to evaluate standardized using the same statistics as the positive class.
    """

    # Fit the scaler on positives:
    scaler = StandardScaler()
    scaler.fit(positive[columns])

    positive_scaled = positive.copy()
    unknown_scaled = unknown.copy()

    # Apply the scaler:
    positive_scaled[columns] = scaler.transform(positive[columns])
    unknown_scaled[columns] = scaler.transform(unknown[columns])

    # Adding back other unrelated columns:
    for col_positive, col_unknown in zip(positive_scaled.columns, unknown_scaled.columns):
        if col_positive not in columns:
            positive_scaled[col_positive] = positive[col_positive].values
        if col_unknown not in columns:
            unknown_scaled[col_unknown] = unknown[col_unknown].values

    return positive_scaled, unknown_scaled


def eval_candidates(unknown: pd.DataFrame,
                    *,
                    n_neighbors: int = 1,
                    score_threshold: int = 1,
                    max_candidates: int | None = None,
                    feature_names: list[str] | None = None,
                    kept_columns: list[str] = ['objectId']
                    ) -> pd.DataFrame:
    """
    Evaluates candidates for the positive class among given objects in the feature space using the nearest neighbors algorithm.  
    The candidates are objects that appear the most in the nearest neighbors of the positive objects.  
    Returns the candidates in a DataFrame with their corresponding score (number of times they appear in the nearest neighbors of the positive objects).

    Parameters
    ---
        unknown: pd.DataFrame
            DataFrame containing the features of all objects to evaluate.

        n_neighbors: int, optional
            Number of neighbors for the nearest neighbors algorithm. Defaults to 1.

        score_threshold: int, optional
            Minimum score for a candidate to be considered. Defaults to 1.

        max_candidates: int | None, optional
            Maximum number of candidates to return. If None, returns all objects. Defaults to None.

        feature_names: list[str] | None, optional
            List of feature names to use for the nearest neighbors algorithm. If None, default feature names are used. Defaults to None.

        kept_columns: list[str], optional
            List of columns to keep in the output DataFrame. Note that if different from default, it should still include 'objectId'. Defaults to ['objectId'].

    Returns
    ---
        candidates: pd.DataFrame
            Candidates for the positive class ordered by the number of times they appear in the nearest neighbors of the positive objects.
    """

    positive = pd.read_parquet(get_mCVs_path().replace('.csv', '_features.parquet'))

    if feature_names is None: # If no feature names are provided, use the default ones:
        feature_names = [
        'amplitude',
        'anderson_darling_normal',
        'beyond_1_std',
        'beyond_2_std',
        'cusum',
        'eta',
        'eta_e',
        'excess_variance',
        'inter_percentile_range_25',
        'inter_percentile_range_10',
        'kurtosis',
        'linear_fit_slope',
        'linear_fit_slope_sigma',
        'linear_fit_reduced_chi2',
        'linear_trend',
        'linear_trend_sigma',
        'linear_trend_noise',
        'magnitude_percentage_ratio_40_5',
        'magnitude_percentage_ratio_20_10',
        'maximum_slope',
        'mean',
        'mean_variance',
        'median',
        'median_absolute_deviation',
        'median_buffer_range_percentage_10',
        'otsu_mean_diff',
        'otsu_std_lower',
        'otsu_std_upper',
        'otsu_lower_to_all_ratio',
        'percent_amplitude',
        'percent_difference_magnitude_percentile_5',
        'percent_difference_magnitude_percentile_20',
        'chi2',
        'roms',
        'skew',
        'standard_deviation',
        'stetson_K',
        'weighted_mean',
        'period_0',
        'period_s_to_n_0',
        'period_1',
        'period_s_to_n_1',
        'period_2',
        'period_s_to_n_2',
        'periodogram_amplitude',
        'periodogram_beyond_1_std',
        'periodogram_beyond_2_std',
        'periodogram_cusum',
        'periodogram_eta',
        'periodogram_inter_percentile_range_25',
        'periodogram_standard_deviation',
        'periodogram_percent_amplitude'
        ]

    # Standardizing the features:
    positive, unknown = fit_scale(positive, unknown, columns=feature_names)

    # Finding the nearest neighbors of positive objects:
    neigh = NearestNeighbors(n_neighbors=n_neighbors).fit(unknown[feature_names])
    neighbors_indices = neigh.kneighbors(positive[feature_names], return_distance=False)
    neighbors = unknown.iloc[neighbors_indices.flatten()]

    # Ids of the neighbors and the number of times each id appears:
    ids, counts = np.unique(neighbors['objectId'], return_counts=True)
    # Scoring the candidates:
    candidates = pd.DataFrame({'objectId': ids, 'score': counts})
    # Zero-score objects:
    zero_ids = unknown[~unknown['objectId'].isin(ids)]['objectId']
    not_candidates = pd.DataFrame({'objectId': zero_ids, 'score': 0})
    # Adding back columns:
    candidates = pd.merge(candidates, unknown[[*kept_columns]], on='objectId', how='left')
    not_candidates = pd.merge(not_candidates, unknown[[*kept_columns]], on='objectId', how='left')
    # output DataFrame:
    out = pd.concat([candidates, not_candidates], ignore_index=True).sort_values(by='score', ascending=False).reset_index(drop=True)

    if max_candidates is None:
        return out[out['score'] >= score_threshold] # Returning only the candidates with a score above the threshold.
    else:
        out = out.iloc[:max_candidates] # Returning only the first 'max_candidates' candidates.
        return out[out['score'] >= score_threshold]
