import pandas as pd
import numpy as np
import io
import requests
from sklearn.neighbors import NearestNeighbors
import light_curve as lc
from magcvs_library.utils import tqdm2


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
    for _, row in tqdm2(list(negative_lc.iterrows())): # Remove list() here for slight better efficiency but no progress bar
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
            new_row_g = pd.DataFrame([dict(zip(negative_lc_g.columns, [row['objectId'], round((max(jd_g)-min(jd_g))/365, 2), len(jd_g), jd_g, magpsf_g, sigmapsf_g]))])
            negative_lc_g = pd.concat([negative_lc_g, new_row_g], ignore_index=True)
        if len(jd_r) >= 4:
            new_row_r = pd.DataFrame([dict(zip(negative_lc_r.columns, [row['objectId'], round((max(jd_r)-min(jd_r))/365, 2), len(jd_r), jd_r, magpsf_r, sigmapsf_r]))])
            negative_lc_r = pd.concat([negative_lc_r, new_row_r], ignore_index=True)

    return negative_lc_g, negative_lc_r


def extract_features(light_curve_data: pd.DataFrame, return_names: bool = False) -> pd.DataFrame:
    """
    Extracts statistical features from light curve data using the light_curve library.

    Parameters
    ---
        light_curve_data: pd.DataFrame
            A DataFrame containing light curve data with columns 'i:jd', 'i:magpsf', and 'i:sigmapsf'.  
            Each row should represent the light curve data for a single object in a single band.  
            Light curve data should be sorted by ascending Julian date and should not contain any missing values.

        return_names: bool, optional
            If True, also returns the names of the extracted features. Defaults to False.

    Returns
    ---
        output_df: pd.DataFrame
            A DataFrame containing the extracted features.  
            It will also keep all columns of the input DataFrame except time, magnitude and error.

        feature_names: list[str], optional
            A list of the names of the extracted features.
    """

    # Initializing features with the light_curve library:
    mean = lc.Mean()
    weighted_mean = lc.WeightedMean()
    standard_deviation = lc.StandardDeviation()
    median = lc.Median()
    amplitude = lc.Amplitude()
    beyond_1_std = lc.BeyondNStd(nstd=1)
    cusum = lc.Cusum()
    inter_percentile_range_10 = lc.InterPercentileRange()
    kurtosis = lc.Kurtosis()
    linear_trend = lc.LinearTrend()
    linear_fit_slope = lc.LinearFit()
    magnitude_percentage_ratio_40_5 = lc.MagnitudePercentageRatio(quantile_numerator=.4, quantile_denominator=.05)
    magnitude_percentage_ratio_20_10 = lc.MagnitudePercentageRatio(quantile_numerator=.2, quantile_denominator=.1)
    maximum_slope = lc.MaximumSlope()
    median_absolute_deviation = lc.MedianAbsoluteDeviation()
    median_buffer_range_percentage_10 = lc.MedianBufferRangePercentage(quantile=.1)
    percent_amplitude = lc.PercentAmplitude()
    mean_variance = lc.MeanVariance()
    anderson_darling_normal = lc.AndersonDarlingNormal()
    chi2 = lc.ReducedChi2()
    skew = lc.Skew()
    stetson_K = lc.StetsonK()
    eta = lc.Eta()
    eta_e = lc.EtaE()
    excess_var = lc.ExcessVariance()
    otsu_split = lc.OtsuSplit()
    percent_diff_mag_percentile = lc.PercentDifferenceMagnitudePercentile()
    robust_median_statistic = lc.Roms()
    beyond_2_std = lc.BeyondNStd(nstd=2)
    # The three features below are experimental, conflicts with extractor.names, might implement in the future:
    #fluxN_not_det_before_Fd = lc.FluxNNotDetBeforeFd()
    #magN_not_det_before_Fd = lc.MagnitudeNNotDetBeforeFd()
    #ptp_var = lc.PeakToPeakVar()

    extractor = lc.Extractor(mean, weighted_mean, standard_deviation, median, amplitude, beyond_1_std,
                            cusum, inter_percentile_range_10, kurtosis, linear_trend, linear_fit_slope,
                            magnitude_percentage_ratio_40_5, magnitude_percentage_ratio_20_10, maximum_slope,
                            median_absolute_deviation, median_buffer_range_percentage_10, percent_amplitude,
                            mean_variance, anderson_darling_normal, chi2, skew, stetson_K,
                            eta, eta_e, excess_var, otsu_split,
                            percent_diff_mag_percentile, robust_median_statistic, beyond_2_std)
    feature_names = extractor.names    

    # Extracting the features for each object in the light curve data:
    features = []
    for line in tqdm2(range(len(light_curve_data))):
        features.append(extractor(light_curve_data['i:jd'][line],
                                  light_curve_data['i:magpsf'][line],
                                  light_curve_data['i:sigmapsf'][line],
                                  sorted=True, check=False)) # Here, sorted and check are set to True and False resp. but they could be passed as kwargs in the future for more flexibility.

    output_df = light_curve_data.drop(columns=['i:jd', 'i:magpsf', 'i:sigmapsf'])
    output_df[feature_names] = np.vstack(features)

    if return_names:
        return output_df, feature_names
    else:
        return output_df


def find_candidates(positive: pd.DataFrame, feature_space: pd.DataFrame, n_neighbors: int = 3, candidate_threshold: int = 2, max_candidates: int | None = None, feature_names: list[str] | None = None):
    """
    Evaluates candidates for the positive class among given objects in the feature space using the nearest neighbors algorithm on given positive class objects.  
    The candidates are objects that appear more than 'candidate_threshold' times in the nearest neighbors of the positive objects.  
    The inputted DataFrames (positive & feature_space) should contain the same features (feature_names) and have columns 'objectId', 'time_range (yr)', 'nb_of_points', 'class'.  
    Returns the candidates in a DataFrame

    Parameters
    ---
        positive: pd.DataFrame
            DataFrame containing the features of positive class objects to find candidates for.

        feature_space: pd.DataFrame
            DataFrame containing the features of all objects to evaluate.

        n_neighbors: int, optional
            Number of neighbors for the nearest neighbors algorithm. Defaults to 3

        candidate_threshold: int, optional
            Parameter for candidate evaluation. Defaults to 2

        max_candidates: int, optional
            Maximum number of candidates to return. If None, there are no limit. Defaults to None

        feature_names: list[str] | None, optional
            List of feature names to use for the nearest neighbors algorithm. If None, default feature names are used. Defaults to None

    Returns
    ---
        candidates: pd.DataFrame
            Candidates for the positive class ordered by the number of times they appear in the nearest neighbors of the positive objects.
    """

    if feature_names is None: # If no feature names are provided, use the default ones:
        feature_names = ['mean', 'weighted_mean', 'standard_deviation', 'median', 'amplitude', 'beyond_1_std', 'cusum', 'inter_percentile_range_25', 'kurtosis',
                         'linear_trend', 'linear_trend_sigma', 'linear_trend_noise', 'linear_fit_slope', 'linear_fit_slope_sigma', 'linear_fit_reduced_chi2',
                         'magnitude_percentage_ratio_40_5', 'magnitude_percentage_ratio_20_10', 'maximum_slope', 'median_absolute_deviation',
                         'median_buffer_range_percentage_10', 'percent_amplitude', 'mean_variance', 'anderson_darling_normal', 'chi2', 'skew', 'stetson_K',
                         'eta', 'eta_e', 'excess_variance', 'otsu_mean_diff', 'otsu_std_lower', 'otsu_std_upper', 'otsu_lower_to_all_ratio',
                         'percent_difference_magnitude_percentile_5', 'roms', 'beyond_2_std']


    # Finding the nearest neighbors of positive objects:
    neigh = NearestNeighbors(n_neighbors=n_neighbors).fit(feature_space[feature_names])
    neighbors_indices = neigh.kneighbors(positive[feature_names], return_distance=False)
    neighbors = feature_space.iloc[neighbors_indices.flatten()]

    # Ids of the neighbors and the number of times each id appears:
    ids, counts = np.unique(neighbors['objectId'], return_counts=True)

    # Creating the DataFrame that will store the candidates:
    candidates = pd.DataFrame(columns=[*feature_space.keys(), 'score'])

    # Adding the candidates to the DataFrame:
    for id, score in zip(ids, counts):
        if score > candidate_threshold: # An object is considered as a candidate if it appears more than 'candidate_threshold' times in the neighbors
            # Put the candidate in the candidates DataFrame with its associated score:
            candidate = feature_space[feature_space['objectId'] == id].iloc[0]
            candidate['score'] = score
            candidates = pd.concat([candidates, candidate.to_frame().T], ignore_index=True)

    # Sorting the candidates by the number of times they appear in the neighbors:
    candidates = candidates.sort_values(by='score', ascending=False).reset_index(drop=True)

    if max_candidates is None:
        return candidates
    else:
        return candidates.iloc[:max_candidates] # Returning only the first 'max_candidates' candidates.