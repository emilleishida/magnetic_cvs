import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from .utils import tqdm2, single_object_lc_data, extract_features, fit_scale
from .managemcvs import get_data_path


def get_mcvs_lightcurve_data(cut: int = 100) -> pd.DataFrame:
    """
    Get mCVs lightcurve data from the mCVs dataset using Fink API. Objects tagged as bogus are excluded and lightcurves are merged for objects with 2 ids.

    Parameters
    ---
        cut: int, optional
            Quality cut for the number of points in the lightcurve. Defaults to 100, meaning only lightcurves with at least 100 points (considering the two bands) will be considered.

    Returns
    ---
        lightcurve_data: pd.Dataframe
            DataFrame containing the lightcurve data for mCVs with columns: 'objectId', 'time_range (yr)' (the time range between first and last detection), 'nb_of_points' (the number of points in the lightcurve), 'i:jd', 'i:magpsf', 'i:sigmapsf'.
    """

    # Load the current mCVs dataset:
    mCVs = pd.read_csv(get_data_path('mCVs.csv'))
    
    # Iterate over the mCVs and retrieve lightcurves:
    rows = []
    for _, row in tqdm2(mCVs.iterrows(), desc='Retrieving lightcurves with Fink API', total=len(mCVs)):
        if row['is_bogus']: # Skip bogus lightcurves.
            continue

        if not isinstance(row['objectId2'], str): # If the object has only one id, retrieve its lightcurve:
            data = single_object_lc_data(row['objectId'])
        else: # If the object has two ids, retrieve and merge both lightcurves:
            data1 = single_object_lc_data(row['objectId'])
            data2 = single_object_lc_data(row['objectId2'])
            t = np.concatenate((data1['i:jd'].values, data2['i:jd'].values))
            m = np.concatenate((data1['i:magpsf'].values, data2['i:magpsf'].values))
            s = np.concatenate((data1['i:sigmapsf'].values, data2['i:sigmapsf'].values))
            idx = np.argsort(t) # Sort by ascending julian date.
            data = pd.DataFrame({
                'i:jd': t[idx],
                'i:magpsf': m[idx],
                'i:sigmapsf': s[idx]
            })

        # Add lightcurve data of current mCV as one row for the output DataFrame:
        rows.append({
                'objectId': row['objectId'],
                'time_range (yr)': round((np.max(data['i:jd'].values) - np.min(data['i:jd'].values)) / 365, 3),
                'nb_of_points': len(data['i:jd'].values),
                'i:jd': data['i:jd'].values,
                'i:magpsf': data['i:magpsf'].values,
                'i:sigmapsf': data['i:sigmapsf'].values
        })

    # Put all rows into a DataFrame:
    lightcurve_data = pd.DataFrame(rows)

    return lightcurve_data[lightcurve_data['nb_of_points'] >= cut].reset_index(drop=True)


def rank_features_by_mutual_info(feature_names: list[str]) -> None:
    """
    Rank features based on mutual information between features and class labels. The ranking is saved and used by the algorithm in the eval_candidates function.

    Parameters
    ---
        feature_names: list[str]
            List of features to consider.
    """

    # Load positive and negative features:
    positive = pd.read_parquet(get_data_path('mCVs_features.parquet'))
    try:
        negative = pd.read_parquet(get_data_path('negative_features.parquet'))
    except FileNotFoundError:
        raise FileNotFoundError(
            'Negative features file not found. Please download the file at https://zenodo.org/communities/fink/TEMPORARY-LINK-NEGATIVE-FEATURES and save it as "negative_features.parquet" under the data directory of this package. Run `mcvs.get_data_path()` to find where the directory is on your computer.'
        )

    # For potentially better dimensionality reduction, compute again negative features on 1-year lightcurve but with only classified objects that are not mCVs (current negative features data is from many objects that are classified or unknown). Having more RRLyrae and Miras for example in these negatives may help so that the algorithm can better make the distinction between these and mCVs.

    # Scale features:
    positive_scaled, negative_scaled = fit_scale(positive, negative, columns=feature_names)

    # Combine data and labels:
    X_all = np.vstack([positive_scaled[feature_names], negative_scaled[feature_names]])
    y_all = np.concatenate([
        np.ones(len(positive), dtype=int),
        np.zeros(len(negative), dtype=int)
    ])

    # MIC can take some time to run (~10s), keeping the user waiting:
    print('Ranking features...')

    # Compute mutual information:
    mi_scores = mutual_info_classif(X_all, y_all, discrete_features=False, random_state=42)

    # Return ranked features:
    feature_scores = pd.DataFrame({
        'feature': feature_names,
        'mutual_information': mi_scores
    }).sort_values(by='mutual_information', ascending=False)

    # Save to csv:
    feature_scores.to_csv(get_data_path('feature_scores.csv'), index=False)

    return


def update_mCVs_features(return_features: bool = False) -> None | pd.DataFrame:
    """
    Compute and save statistical features for mCVs lightcurves using the Fink API and the light_curve library.
    
    Parameters
    ---
        return_features: bool, optional
            If True, returns the DataFrame with the extracted features. Defaults to False.

    Returns
    ---
        mCVs_features: pd.DataFrame, optional
            DataFrame containing the extracted features for mCVs lightcurves. Only returned if `return_features` is True.
    """

    # Load mCVs lightcurves
    mCVs_lightcurves = get_mcvs_lightcurve_data()

    # Extract features
    mCVs_features, feature_names = extract_features(mCVs_lightcurves, return_names=True)

    # Save features to parquet:
    mCVs_features.to_parquet(get_data_path('mCVs_features.parquet'), index=False)

    # Run mutual info classifier and rank features for dimensionality reduction before eval_candidates:
    rank_features_by_mutual_info(feature_names)

    print('mCVs features updated successfully.')

    if return_features:
        return mCVs_features
    else:
        return


