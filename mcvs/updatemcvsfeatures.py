import pandas as pd
import numpy as np
from .utils import tqdm2, lc_data_from_api, extract_all_features
from .managemcvs import get_mCVs_path


def get_mcvs_lightcurve_data(cut: int = 100) -> pd.DataFrame:
    """
    Get mCVs lightcurve data from the mCVs dataset using Fink API. Objects tagged as bogus are excluded and lightcurves are merged for objects with 2 ids.

    Parameters
    ---
        cut: int
            Quality cut for the number of points in the lightcurve. Defaults to 100, meaning only lightcurves with at least 100 points (considering the two bands) will be considered.

    Returns
    ---
        lightcurve_data: pd.Dataframe
            DataFrame containing the lightcurve data for mCVs with columns: 'objectId', 'time_range (yr)' (the time range between first and last detection), 'nb_of_points' (the number of points in the lightcurve), 'i:jd', 'i:magpsf', 'i:sigmapsf'.
    """

    # Load the current mCVs dataset:
    mCVs = pd.read_csv(get_mCVs_path())
    
    # Iterate over the mCVs and retrieve lifghtcurves:
    rows = []
    for _, row in tqdm2(mCVs.iterrows(), desc='Retrieving lightcurves with Fink API', total=len(mCVs)):
        if row['is_bogus']: # Skip bogus lightcurves.
            continue

        if not isinstance(row['objectId2'], str): # If the object has only one id, retrieve its lightcurve:
            data = lc_data_from_api(row['objectId'])
        else: # If the object has two ids, retrieve and merge both lightcurves:
            data1 = lc_data_from_api(row['objectId'])
            data2 = lc_data_from_api(row['objectId2'])
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


def update_mCVs_features(return_features: bool = False) -> None | pd.DataFrame:
    """
    Compute and save statistical features for mCVs lightcurves using the Fink API and the light_curve library.
    
    Parameters
    ---
        return_features: bool, optional
            If True, returns the DataFrame with the extracted features. Defaults to False.

    Returns
    ---
        mCVs_features: pd.DataFrame
            DataFrame containing the extracted features for mCVs lightcurves. Only returned if `return_features` is True.
    """

    mCVs_lightcurves = get_mcvs_lightcurve_data()

    mCVs_features = extract_all_features(mCVs_lightcurves)

    mCVs_features.to_parquet(get_mCVs_path().replace('.csv', '_features.parquet'), index=False)

    print('mCVs features updated successfully.')

    if return_features:
        return mCVs_features
    else:
        return


