import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .utils import fit_scale, get_data_path, many_objects_lc_data, extract_features


def eval_candidates(unknown: pd.DataFrame,
                    *,
                    n_neighbors: int = 1,
                    score_threshold: int = 0,
                    max_candidates: int | None = None,
                    feature_names: list[str] | str | None = None,
                    kept_columns: list[str] = ['objectId']
                    ) -> pd.DataFrame:
    """
    Evaluates candidates for the positive class among given objects in the feature space using the nearest neighbors algorithm.  
    The candidates are objects that appear the most in the nearest neighbors of the positive objects.  
    Returns the candidates in a DataFrame with their corresponding score (number of times they appear in the nearest neighbors of the positive objects).

    Parameters
    ---
        unknown: pd.DataFrame
            DataFrame containing the features of all objects to evaluate. One object per row, one feature per column.

        n_neighbors: int, optional
            Number of neighbors for the nearest neighbors algorithm. Defaults to 1.

        score_threshold: int, optional
            Minimum score for a candidate to be considered. Defaults to 0, meaning all objects are returned.

        max_candidates: int | None, optional
            Maximum number of candidates to return. If None, returns all objects with a score higher or equal to `score_threshold`. Defaults to None.

        feature_names: list[str] | str | None, optional
            List of feature names to use for the nearest neighbors algorithm. Also accepts the value 'all' in which case all features (check all features list in the `extract_features` definition) will be considered. If None, default feature names are used based on mutual information scores. Defaults to None.

        kept_columns: list[str], optional
            List of columns to keep in the output DataFrame. Note that if different from default, it should still include 'objectId'. Defaults to ['objectId'].

    Returns
    ---
        candidates: pd.DataFrame
            Candidates for the positive class ordered by the number of times they appear in the nearest neighbors of the positive objects.
    """

    positive = pd.read_parquet(get_data_path('mCVs_features.parquet'))

    if feature_names is None: # If no feature names are provided, use the default ones:
        feature_names = pd.read_csv(get_data_path('feature_scores.csv'))['feature'].tolist()[:20] # Taking the first 20 out of the 52 features is an arbitrary choice here. Dimensionality reduction was explored, but no significative performance improvement was observed. This is due to the poverty of the positive dataset. With more data and with higher quality, one could rework the dimensionality reduction or even think of a better algorithm in order to improve performances.
    elif feature_names == 'all':
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


def eval_distance(objectIds: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Evaluate (euclidean) distances to the center distribution of mCVs for given objects.  
    **/!\\ This function uses the Fink API to concatenate full lightcurves. It is thus not designed for large queries.**  
    This function is intended to be used on high-score objects obtained with the `eval_candidates` function, allowing to have more precise information on how likely candidates are bona-fide mCVs according to the current state of the algorithm.

    Parameters
    ---
        objectIds: list[str]
            List of ZTF objectIds for which to evaluate their distance to the mCVs center distribution in the feature space.

    Returns
    ---
        candidates: pd.DataFrame
            DataFrame containing given candidates sorted by increasing distance to the mCVs center distribution.

        mCVs_statistics: pd.Series
            Distance statistics of the mCVs set for comparison.
    """

    # ---
    # Potential future improvements:
    # Use the SNAD API instead of Fink API to get augmented data (from data release) for the objects to be evaluated AND the mCVs (so features should be computed once again but with this augmented data).
    # Plus, use dimensionality reduction.
    # ---

    # Load mCVs feature data:
    mCVs_features = pd.read_parquet(get_data_path('mCVs_features.parquet'))

    # Get full lightcurves of given objects:
    unknown_lightcurves = many_objects_lc_data(objectIds)
    # Compute associated features:
    unknown_features, feature_names = extract_features(unknown_lightcurves, return_names=True)

    # Standardizing the features:
    mCVs_features, unknown_features = fit_scale(mCVs_features, unknown_features, columns=feature_names)

    # Compute Euclidean distance from the origin (center of the mCVs distribution after scaling):
    unknown_distances = np.linalg.norm(unknown_features[feature_names], axis=1) #/ np.sqrt(len(feature_names))
    mCVs_distances = np.linalg.norm(mCVs_features[feature_names], axis=1) #/ np.sqrt(len(feature_names))

    # Add distances and remove feature columns:
    candidates = unknown_features.drop(columns=feature_names).copy()
    candidates['distance'] = unknown_distances
    mCVs = mCVs_features.drop(columns=feature_names).copy()
    mCVs['distance'] = mCVs_distances

    # Sort by distance and return:
    return candidates.sort_values(by='distance', ascending=True).reset_index(drop=True), mCVs['distance'].describe()


