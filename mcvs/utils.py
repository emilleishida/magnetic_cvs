import pandas as pd
import numpy as np
import io
import requests
from tqdm import tqdm
import light_curve as lc


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
def lc_data_from_api(objectId: str) -> pd.DataFrame:
    """
    Get lightcurve data (both g and r bands considered) for a given objectId from the Fink API.

    Parameters
    ---
        objectId: str
            The ZTF object ID (format: ZTFXXaaaaaa) for which to retrieve the lightcurve data.

    Returns
    ---
        lightcurve_data: pd.DataFrame
            Corresponding lightcurve data with columns: 'objectId', 'i:jd', 'i:magpsf', 'i:sigmapsf'.
    """

    lightcurve_data = pd.read_json(io.BytesIO(requests.post("https://api.fink-portal.org/api/v1/objects",
                                                json={"objectId": objectId, "columns": "i:objectId,i:jd,i:magpsf,i:sigmapsf", "output-format": "json"}
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
        lc.WeightedMean(),
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
                                          lc.PercentAmplitude()]))

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


