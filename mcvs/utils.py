import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import io
import requests
from tqdm import tqdm
import light_curve as lc


def get_data_path(file: str = '') -> str:
    """
    Get the absolute path to the data directory of this package.
    
    Parameters
    ---
        file: str, optional
            The file name to be appended at the end of the path.
    
    Returns
    ---
        data_path: str
            The absolute path to the data directory of this package as a string.
    """
    return str(Path(__file__).parent / 'data') + '/' + file


# Wrapper for tqdm with default arguments:
def tqdm2(iterable,
          **kwargs
          ) -> tqdm:
    """
    Modified tqdm function with lighter default arguments.
    
    tqdm is a library for creating progress bars in Python.

    Parameters
    ---
        iterable:
            The iterable to be wrapped with a progress bar.
        kwargs:
            Additional keyword arguments to be passed to tqdm.
    """

    default_kwargs = {'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt}'}
    default_kwargs.update(kwargs)

    return tqdm(iterable, **default_kwargs)

# Wrapper for Fink API to get lightcurve data in a pandas DataFrame:
def lc_data_from_api(objectId: str,
                     columns: str = "i:objectId,i:jd,i:magpsf,i:sigmapsf"
                     ) -> pd.DataFrame:
    """
    Get lightcurve data (both g and r bands considered) for a given objectId from the Fink API.

    Parameters
    ---
        objectId: str
            The ZTF object ID (format: ZTFXXaaaaaa) for which to retrieve the lightcurve data.

        columns: str, optional
            The columns to be retrieved from the Fink API. Defaults to "i:objectId,i:jd,i:magpsf,i:sigmapsf".

    Returns
    ---
        lightcurve_data: pd.DataFrame
            Corresponding lightcurve data with columns: 'objectId', 'i:jd', 'i:magpsf', 'i:sigmapsf'.
    """

    lightcurve_data = pd.read_json(io.BytesIO(requests.post("https://api.fink-portal.org/api/v1/objects",
                                                json={"objectId": objectId, "columns": columns, "output-format": "json"}
                                                ).content
                                )
                    ).sort_values(by='i:jd') # Sorting by ascending julian date for the feature extractor (default output is descending).

    return lightcurve_data


def extract_features(light_curve_data: pd.DataFrame,
                     return_names: bool = False,
                     **kwargs
                     ) -> pd.DataFrame:
    """
    Extracts statistical features from light curve data using the light_curve library.

    Parameters
    ---
        light_curve_data: pd.DataFrame
            A DataFrame containing lightcurve data with columns 'i:jd', 'i:magpsf', and 'i:sigmapsf'.  
            Each row should represent the lightcurve data for a single object.

        return_names: bool, optional
            If True, also returns the names of the extracted features. Defaults to False.

        **kwargs: optional
            Additional keyword arguments to be passed to the light_curve extractor. Default arguments are sorted=True and check=False, supposing the light curve data is sorted by ascending Julian date and does not contain any missing values.

    Returns
    ---
        output_df: pd.DataFrame
            A DataFrame containing the extracted features. It will also keep all columns of the input DataFrame except time, magnitude and error.

        feature_names: list[str], optional
            A list of names of the extracted features.
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
        lc.WeightedMean()
    )
    # Periodogram features require an extractor in which we will not pass the uncertainties:
    extractor2 = lc.Extractor(
        lc.Periodogram(peaks=3, features=[lc.Amplitude(),
                                          lc.BeyondNStd(nstd=1),
                                          lc.BeyondNStd(nstd=2),
                                          lc.Cusum(),
                                          lc.Eta(),
                                          lc.InterPercentileRange(quantile=.25),
                                          lc.StandardDeviation(),
                                          lc.PercentAmplitude()
                                          ]
                       )
        )

    feature_names = extractor1.names + extractor2.names

    # Extracting the features for each object in the lightcurve data:
    features = []
    for _, row in tqdm2(light_curve_data.iterrows(), desc='Extracting features', total=len(light_curve_data)):
        features1 = extractor1(row['i:jd'],
                               row['i:magpsf'],
                               row['i:sigmapsf'],
                               **default_kwargs)
        features2 = extractor2(row['i:jd'],
                               row['i:magpsf'],
                               **default_kwargs)
        features.append(np.append(features1, features2))

    output_df = light_curve_data.drop(columns=['i:jd', 'i:magpsf', 'i:sigmapsf'])
    output_df[feature_names] = np.vstack(features)

    if return_names:
        return output_df, feature_names
    else:
        return output_df


'''
def extract_missing_features(light_curve_data: pd.DataFrame,
                             return_names: bool = False,
                             **kwargs
                             ) -> pd.DataFrame:
    """
    Extracts statistical features **that are missing in Fink alerts** from light curve data using the light_curve library.

    Parameters
    ---
        light_curve_data: pd.DataFrame
            A DataFrame containing lightcurve data with columns 'i:jd', 'i:magpsf', and 'i:sigmapsf'.  
            Each row should represent the lightcurve data for a single object.

        return_names: bool, optional
            If True, also returns the names of the extracted features. Defaults to False.

        **kwargs: optional
            Additional keyword arguments to be passed to the light_curve extractor. Default arguments are sorted=True and check=False, supposing the light curve data is sorted by ascending Julian date and does not contain any missing values.

    Returns
    ---
        output_df: pd.DataFrame
            A DataFrame containing the extracted features. It will also keep all columns of the input DataFrame except time, magnitude and error.

        feature_names: list[str], optional
            A list of names of the extracted features.
    """

    default_kwargs = {'sorted': True, 'check': False}
    default_kwargs.update(**kwargs)

    extractor1 = lc.Extractor(
        lc.BeyondNStd(nstd=2),
        lc.Eta(),
        lc.EtaE(),
        lc.ExcessVariance(),
        lc.InterPercentileRange(quantile=.25),
        lc.OtsuSplit(),
        lc.PercentDifferenceMagnitudePercentile(quantile=.05),
        lc.PercentDifferenceMagnitudePercentile(quantile=.20),
        lc.Roms()
    )
    # Periodogram features require an extractor in which we will not pass the uncertainties:
    extractor2 = lc.Extractor(
        lc.Periodogram(peaks=3, features=[lc.Amplitude(),
                                          lc.BeyondNStd(nstd=1),
                                          lc.BeyondNStd(nstd=2),
                                          lc.Cusum(),
                                          lc.Eta(),
                                          lc.InterPercentileRange(quantile=.25),
                                          lc.StandardDeviation(),
                                          lc.PercentAmplitude()
                                          ]
                       )
        )

    # The features below are experimental, conflicts with extractor.names, might implement in the future:
    # lc.MagnitudeNNotDetBeforeFd()
    # lc.PeakToPeakVar()

    feature_names = extractor1.names + extractor2.names

    # Extracting the features for each object in the lightcurve data:
    features = []
    for _, row in tqdm2(light_curve_data.iterrows(), desc='Extracting features', total=len(light_curve_data)):
        features1 = extractor1(row['i:jd'],
                               row['i:magpsf'],
                               row['i:sigmapsf'],
                               **default_kwargs)
        features2 = extractor2(row['i:jd'],
                               row['i:magpsf'],
                               **default_kwargs)
        features.append(np.append(features1, features2))

    output_df = light_curve_data.drop(columns=['i:jd', 'i:magpsf', 'i:sigmapsf'])
    output_df[feature_names] = np.vstack(features)

    if return_names:
        return output_df, feature_names
    else:
        return output_df
'''


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
    for col in positive.columns:
        if col not in columns:
            positive_scaled[col] = positive[col].values
    for col in unknown.columns:
        if col not in columns:
            unknown_scaled[col] = unknown[col].values

    return positive_scaled, unknown_scaled


def fink_lightcurve(objectId: str) -> None:
    """
    Plot the lightcurve of an object given its Id using the Fink API. Design and color scheme faithful to the visual style of the Fink portal.

    Parameters
    ---
        objectId: str
            The Id of the object to plot. Format: 'ZTF20abcdefg'
    """

    # Get the light curve data of the specified object:
    lightcurve = lc_data_from_api(objectId, columns="i:objectId,i:jd,i:magpsf,i:sigmapsf,i:fid")

    # Extract time, magnitude, magnitude error and filter:
    t = lightcurve['i:jd'].values
    m = lightcurve['i:magpsf'].values
    m_err = lightcurve['i:sigmapsf'].values
    fid = lightcurve['i:fid'].values

    # Convert time values from JD to years:
    t = (t - 2451545.0) / 365.25 + 2000

    # Plot the lightcurve:
    plt.figure(figsize=(14, 6))
    plt.rcParams['font.size'] = 14
    plt.title(objectId)

    plt.errorbar(t[fid == 1], m[fid == 1], yerr=m_err[fid == 1], fmt='o', c='#15284F', label='g band')
    plt.errorbar(t[fid == 2], m[fid == 2], yerr=m_err[fid == 2], fmt='o', c='#F5622E', label='r band')
    plt.xlabel('Year')
    plt.ylabel('Magnitude')
    plt.legend(loc='upper right', ncol=2, facecolor='#D5D5D3', fontsize=10)
    plt.gca().invert_yaxis()
    plt.ylim(m.max() + .5, m.min() - .5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(color='#D5D5D3')

    return


