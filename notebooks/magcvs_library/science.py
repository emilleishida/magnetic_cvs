import pandas as pd
import numpy as np
import io
import requests
from sklearn.neighbors import NearestNeighbors
import light_curve as lc
from .utils import tqdm2


# Wrapper for fink api (to be modified for more options... - look in fink api doc the columns available for json argument):
def get_lightcurve_data(Ids: list[str] | str, cut: int = 4, split_by_filter: bool = True) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    """
    Retreive light curve data using Fink API for a given set of object Ids.

    Columns in the output dataframes:
        - objectId: The ID of the object. (str)
        - time_range (yr): The time range between the first and last observation of the object in years. Not yet returned when split_by_filter is False. (float)
        - nb_of_points: The number of data points in the light curve. Not yet returned when split_by_filter is False. (int)
        - i:jd: Julian date of the observations. (array)
        - i:magpsf: Magnitude of the observations. (array)
        - i:sigmapsf: Magnitude error of the observations. (array)
        - i:fid: Filter ID of the observations. Only returned if split_by_filter is False. 1 for g filter, 2 for r filter. (array)

    Parameters
    ---
        Ids: list[str] | str
            A list containing the object Ids for which to extract light curve data. Can be passed as a string for a single object Id.

        cut: int, optional
            A threshold on the number of points in the light curves. Light curves with less than 'cut' points will not be returned. Defaults to 4.

        split_by_filter: bool, optional
            If True, the light curve data will be split into two separate dataframes for g and r filters. If False, the data will be returned in one dataframe with a filter column. Defaults to True.

    Returns
    ---
        data_g, data_r: pd.DataFrame, pd.DataFrame
            Light curve data for the g and r filters in separated dataframes. Returned if split_by_filter is True.

        data: pd.DataFrame
            Light curve data for both filters in one dataframe. Returned if split_by_filter is False.
    """

    if type(Ids) == str:
        Ids = [Ids]

    if not split_by_filter:
        data = pd.read_json(io.BytesIO(requests.post("https://api.fink-portal.org/api/v1/objects",
                                                    json={"objectId": Ids, "columns": "i:objectId,i:jd,i:magpsf,i:sigmapsf,i:fid", "output-format": "json"}
                                                    ).content
                                      )
                           )
        return data

    # else:
    # Initializing the two dataframes which will contain the light curve data in g and r filter:
    data_g = pd.DataFrame(columns=['objectId', 'time_range (yr)', 'nb_of_points', 'i:jd', 'i:magpsf', 'i:sigmapsf'])
    data_r = pd.DataFrame(columns=['objectId', 'time_range (yr)', 'nb_of_points', 'i:jd', 'i:magpsf', 'i:sigmapsf'])

    for object in tqdm2(Ids):
        # Getting the data from the current object with fink api:
        pdf = pd.read_json(io.BytesIO(requests.post("https://api.fink-portal.org/api/v1/objects",
                                                    json={"objectId": object, "columns": "i:objectId,i:jd,i:magpsf,i:sigmapsf,i:fid", "output-format": "json"}
                                                    ).content
                                      )
                           ).sort_values(by='i:jd') # Sorting by ascending julian date for the extractor. (Default output is descending)
        # g filter:
        pdf_g = pdf[pdf['i:fid'] == 1]
        if len(pdf_g) >= cut: # Extractor does not accept less than 4 data points.
            jd_g = pdf_g['i:jd'].values
            magpsf_g = pdf_g['i:magpsf'].values
            sigmapsf_g = pdf_g['i:sigmapsf'].values
            new_row = pd.DataFrame([dict(zip(data_g.columns, [object, round((max(jd_g)-min(jd_g))/365, 3), len(jd_g), jd_g, magpsf_g, sigmapsf_g]))])
            data_g = pd.concat([data_g, new_row], ignore_index=True)

        # r filter:
        pdf_r = pdf[pdf['i:fid'] == 2]
        if len(pdf_r) >= cut:
            jd_r = pdf_r['i:jd'].values
            magpsf_r = pdf_r['i:magpsf'].values
            sigmapsf_r = pdf_r['i:sigmapsf'].values
            new_row = pd.DataFrame([dict(zip(data_r.columns, [object, round((max(jd_r)-min(jd_r))/365, 3), len(jd_r), jd_r, magpsf_r, sigmapsf_r]))])
            data_r = pd.concat([data_r, new_row], ignore_index=True)

    return data_g, data_r


def sort_negative(negative_lc: pd.DataFrame, positive_Ids: list[str]):
    """
    Given a DataFrame containing the light curve data, removes potential objects that have positive_Ids and returns the data splitted by filter ready for feature extraction. 

    Parameters
    ---
        negative_lc: pd.DataFrame
            A DataFrame containing light curve data with columns 'i:jd', 'i:magpsf', 'i:sigmapsf' and 'i:fid'.  
            Each row should represent the light curve data for a single object.

        positive_Ids: list[str]
            A list of object Ids to be removed from negative_lc.

    Returns
    ---
        negative_lc_g, negative_lc_r: pd.DataFrame, pd.DataFrame
            Light curve data for the g and r filters in separated dataframes.  
            Added columns: 'time_range (yr)' and 'nb_of_points', which correspond to the time range between the first and last observation of the object in years and the number of data points in the light curve respectively.
    """

    # Removing potential positive class objects from the negative class:
    negative_Ids = negative_lc['objectId'].values
    intersect = np.intersect1d(negative_Ids, positive_Ids)
    negative_lc = negative_lc[~np.isin(negative_Ids, intersect)]
    
    # Splitting data by filter:
    negative_lc_g = pd.DataFrame(columns=['objectId','time_range (yr)', 'nb_of_points', 'i:jd', 'i:magpsf', 'i:sigmapsf'])
    negative_lc_r = pd.DataFrame(columns=['objectId','time_range (yr)', 'nb_of_points', 'i:jd', 'i:magpsf', 'i:sigmapsf'])
    for _, row in tqdm2(list(negative_lc.iterrows()), desc='Splitting data by filter'): # Remove list() here for slight better efficiency but no progress bar
        jd_g, magpsf_g, sigmapsf_g = np.array([]), np.array([]), np.array([])
        jd_r, magpsf_r, sigmapsf_r = np.array([]), np.array([]), np.array([])
        for index, fid in enumerate(row['i:fid']):
            if fid == 1:
                jd_g = np.append(jd_g, row['i:jd'][index])
                magpsf_g = np.append(magpsf_g, row['i:magpsf'][index])
                sigmapsf_g = np.append(sigmapsf_g, row['i:sigmapsf'][index])
            elif fid == 2:
                jd_r = np.append(jd_r, row['i:jd'][index])
                magpsf_r = np.append(magpsf_r, row['i:magpsf'][index])
                sigmapsf_r = np.append(sigmapsf_r, row['i:sigmapsf'][index])
        if len(jd_g) >= 4:
            sorted_indices_g = np.argsort(jd_g)
            new_row_g = pd.DataFrame([dict(zip(negative_lc_g.columns,
                                               [
                                                   row['objectId'],
                                                   round((max(jd_g)-min(jd_g))/365, 3),
                                                   len(jd_g),
                                                   jd_g[sorted_indices_g],
                                                   magpsf_g[sorted_indices_g],
                                                   sigmapsf_g[sorted_indices_g]
                                                ]))])
            negative_lc_g = pd.concat([negative_lc_g, new_row_g], ignore_index=True)
        if len(jd_r) >= 4:
            sorted_indices_r = np.argsort(jd_r)
            new_row_r = pd.DataFrame([dict(zip(negative_lc_r.columns,
                                               [
                                                   row['objectId'],
                                                   round((max(jd_r)-min(jd_r))/365, 3),
                                                   len(jd_r),
                                                   jd_r[sorted_indices_r],
                                                   magpsf_r[sorted_indices_r],
                                                   sigmapsf_r[sorted_indices_r]
                                                ]))])
            negative_lc_r = pd.concat([negative_lc_r, new_row_r], ignore_index=True)

    return negative_lc_g, negative_lc_r


def extract_features(light_curve_data: pd.DataFrame, return_names: bool = False, **kwargs) -> pd.DataFrame:
    """
    Extracts statistical features from light curve data using the light_curve library.

    Parameters
    ---
        light_curve_data: pd.DataFrame
            A DataFrame containing light curve data with columns 'i:jd', 'i:magpsf', and 'i:sigmapsf'.  
            Each row should represent the light curve data for a single object in a single band.

        return_names: bool, optional
            If True, also returns the names of the extracted features. Defaults to False.

        **kwargs: optional
            Additional keyword arguments to be passed to the light_curve extractor. Default arguments are sorted=True and check=False,
            supposing the light curve data is sorted by ascending Julian date and does not contain any missing values.

    Returns
    ---
        output_df: pd.DataFrame
            A DataFrame containing the extracted features.  
            It will also keep all columns of the input DataFrame except time, magnitude and error.

        feature_names: list[str], optional
            A list of the names of the extracted features.
    """

    default_kwargs = {'sorted': True, 'check': False}
    default_kwargs.update(**kwargs)
    
    extractor1 = lc.Extractor(
        lc.Amplitude(),
        lc.AndersonDarlingNormal(),
        lc.BeyondNStd(nstd=1),
        lc.BeyondNStd(nstd=2),
        lc.Cusum(),
        lc.Eta(),
        lc.EtaE(),
        lc.ExcessVariance(),
        lc.InterPercentileRange(quantile=.25),
        lc.InterPercentileRange(quantile=.10),
        lc.Kurtosis(),
        lc.LinearFit(),
        lc.LinearTrend(),
        lc.MagnitudePercentageRatio(quantile_numerator=.4, quantile_denominator=.05),
        lc.MagnitudePercentageRatio(quantile_numerator=.2, quantile_denominator=.1),
        lc.MaximumSlope(),
        lc.Mean(),
        lc.MeanVariance(),
        lc.Median(),
        lc.MedianAbsoluteDeviation(),
        lc.MedianBufferRangePercentage(quantile=.1),
        lc.OtsuSplit(),
        lc.PercentAmplitude(),
        lc.PercentDifferenceMagnitudePercentile(quantile=.05),
        lc.PercentDifferenceMagnitudePercentile(quantile=.20),
        lc.ReducedChi2(),
        lc.Roms(),
        lc.Skew(),
        lc.StandardDeviation(),
        lc.StetsonK(),
        lc.WeightedMean(),
    )
    # Periodogram requires an extractor in which we will not pass the uncertainties:
    extractor2 = lc.Extractor(
        lc.Periodogram(peaks=3, features=[lc.Amplitude(),
                                          lc.BeyondNStd(nstd=1),
                                          lc.BeyondNStd(nstd=2),
                                          lc.Cusum(),
                                          lc.Eta(),
                                          lc.InterPercentileRange(quantile=.25),
                                          lc.StandardDeviation(),
                                          lc.PercentAmplitude()]))

    # The features below are experimental, conflicts with extractor.names, might implement in the future:
    #lc.FluxNNotDetBeforeFd()
    #lc.MagnitudeNNotDetBeforeFd()
    #lc.PeakToPeakVar()

    feature_names = extractor1.names + extractor2.names

    # Extracting the features for each object in the light curve data:
    features = []
    for line in tqdm2(range(len(light_curve_data)), desc='Extracting features'):
        features1 = extractor1(light_curve_data['i:jd'][line],
                               light_curve_data['i:magpsf'][line],
                               light_curve_data['i:sigmapsf'][line],
                               **default_kwargs)

        features2 = extractor2(light_curve_data['i:jd'][line],
                               light_curve_data['i:magpsf'][line],
                               **default_kwargs)

        features.append(np.append(features1, features2))

    output_df = light_curve_data.drop(columns=['i:jd', 'i:magpsf', 'i:sigmapsf'])
    output_df[feature_names] = np.vstack(features)

    if return_names:
        return output_df, feature_names
    else:
        return output_df


def find_candidates(positive: pd.DataFrame,
                    feature_space: pd.DataFrame,
                    *,
                    n_neighbors: int = 3,
                    score_threshold: int = 1,
                    max_candidates: int | None = 5,
                    feature_names: list[str] | None = None,
                    kept_columns: list[str] = ['objectId']
                    ) -> pd.DataFrame:
    """
    Evaluates candidates for the positive class among given objects in the feature space using the nearest neighbors algorithm on given positive class objects.  
    The candidates are objects that appear the most in the nearest neighbors of the positive objects.  
    The inputted DataFrames (positive & feature_space) should contain the same features (feature_names) and have the column 'objectId'.  
    Returns the candidates in a DataFrame with their corresponding score (number of times they appear in the nearest neighbors of the positive objects).

    Parameters
    ---
        positive: pd.DataFrame
            DataFrame containing the features of positive class objects to find candidates for.

        feature_space: pd.DataFrame
            DataFrame containing the features of all objects to evaluate.

        n_neighbors: int, optional
            Number of neighbors for the nearest neighbors algorithm. Defaults to 3.

        score_threshold: int, optional
            Minimum score for a candidate to be considered. Defaults to 1.

        max_candidates: int | None, optional
            Maximum number of candidates to return. If None, returns all the nearest neighbors of the positive objects. Defaults to 5.

        feature_names: list[str] | None, optional
            List of feature names to use for the nearest neighbors algorithm. If None, default feature names are used. Defaults to None.

        kept_columns: list[str], optional
            List of columns to keep in the output DataFrame. Note that if different from default, it should still include 'objectId'. Defaults to ['objectId'].

    Returns
    ---
        candidates: pd.DataFrame
            Candidates for the positive class ordered by the number of times they appear in the nearest neighbors of the positive objects.
    """

    if feature_names is None: # If no feature names are provided, use the default ones:
        feature_names = ['mean', 'weighted_mean', 'standard_deviation', 'median', 'amplitude',
       'beyond_1_std', 'cusum', 'inter_percentile_range_25', 'kurtosis',
       'linear_trend', 'linear_trend_sigma', 'linear_trend_noise',
       'linear_fit_slope', 'linear_fit_slope_sigma', 'linear_fit_reduced_chi2',
       'magnitude_percentage_ratio_40_5', 'magnitude_percentage_ratio_20_10',
       'maximum_slope', 'median_absolute_deviation',
       'median_buffer_range_percentage_10', 'percent_amplitude',
       'mean_variance', 'anderson_darling_normal', 'chi2', 'skew', 'stetson_K',
       'eta', 'eta_e', 'excess_variance', 'otsu_mean_diff', 'otsu_std_lower',
       'otsu_std_upper', 'otsu_lower_to_all_ratio',
       'percent_difference_magnitude_percentile_5', 'roms', 'beyond_2_std',
       'period_0', 'period_s_to_n_0', 'periodogram_amplitude']

    # Finding the nearest neighbors of positive objects:
    neigh = NearestNeighbors(n_neighbors=n_neighbors).fit(feature_space[feature_names])
    neighbors_indices = neigh.kneighbors(positive[feature_names], return_distance=False)
    neighbors = feature_space.iloc[neighbors_indices.flatten()]

    # Ids of the neighbors and the number of times each id appears:
    ids, counts = np.unique(neighbors['objectId'], return_counts=True)

    candidates = pd.DataFrame({'objectId': ids, 'score': counts}).sort_values(by='score', ascending=False).reset_index(drop=True)

    # Adding the time range, number of points and class to the candidates:
    candidates = pd.merge(candidates, feature_space[[*kept_columns]], on='objectId', how='left')
    # Removing duplicates:
    candidates = candidates.drop_duplicates(subset=['objectId'], keep='first').reset_index(drop=True)

    if max_candidates is None:
        return candidates[candidates['score'] >= score_threshold] # Returning only the candidates with a score above the threshold.
    else:
        candidates = candidates.iloc[:max_candidates] # Returning only the first 'max_candidates' candidates.
        return candidates[candidates['score'] >= score_threshold]