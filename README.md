# Magnetic Cataclysmic Variable Stars
Fink science module to find magnetic cataclysmic variable stars. Algorithm based on statistical features computed with the [light-curve package](https://github.com/light-curve/light-curve-python) from [Malanchev et al., 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.5147M/abstract).

## 🔧 Installation Guide

1. Clone the repository:
   ```bash
   git clone https://github.com/emilleishida/mcvs.git
   cd mcvs
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package locally:
   ```bash
   pip install .
   ```

5. (Optional) Download data:  
   If you find new mCVs and wish to [update the set](#manage-mcvs-base-set) used as a base for the algorithm in this package in order to make it more representative, you will need to download [additional data from zenodo](https://zenodo.org/communities/fink/TEMPORARY-LINK-NEGATIVE-FEATURES). The file should be saved as `"negative_features.parquet"` under the `data` directory of this package. In order to find where that directory is, run:
   ```python
   import mcvs

   mcvs.utils.get_data_path()
   ```

## 🌌 Usage Examples

### Evaluate Candidates

1. Load your lightcurve DataFrame of objects to be evaluated.  
   Each row should represent the lightcurve data for a single object, and the DataFrame should contain columns `'objectId'`, `'i:jd'`, `'i:magpsf'` and `'i:sigmapsf'`.  
   Each lightcurve data should be sorted by ascending Julian date and should not contain any missing values. If that is not the case, check the keyword arguments of the `mcvs.extract_features()` function in step 2.
   ```python
   import mcvs
   import pandas as pd

   unknown_lightcurves_df = pd.read_parquet('PATH/TO/YOUR/FILE')
   ```

2. Then compute the associated features:
   ```python
   unknown_features_df = mcvs.extract_features(unknown_lightcurves_df)
   ```

3. And then run the algorithm to evaluate candidates:
   ```python
   candidates = mcvs.eval_candidates(unknown_features_df)
   candidates
   ```

### Manage mCVs base set

⚠️ Warning: All modifications of the mCVs set that are shown in this section will **not** update the base for the algorithm until `mcvs.update_mCVs_features()` is run. So remember to run this once you are done with all your modifications.

+ To view the current set of mCVs, run the following:
   ```python
   import mcvs
   import pandas as pd

   mcvs_df = pd.read_csv(mcvs.get_data_path('mCVs.csv'))
   mcvs_df
   ```

+ You found a new mCV and wish to save it? Use `mcvs.add_new_mCVs('YOUR_NEW_MCV_ZTF_ID')`.  

+ Made a mistake? Use `mcvs.remove_mCVs('MCV_ZTF_ID_TO_BE_REMOVED')`.  

+ You found the Fink lightcurve of one mCV is bogus or not representative of its class? Tag it with `mcvs.tag_bogus('BOGUS_MCV_ZTF_ID')`. This way, it will not be considered in the algorithm.

+ Polars can also be tagged with `mcvs.tag_polars('POLAR_MCV_ZTF_ID')`. In the current state of the algorithm, polar tags do not change anything. But in the future, if enough polars and intermediate polars are discovered so that their feature distributions are significatively different, The algorithm could be splitted in two parts: one for polars and one for intermediate polars.

### Lightcurve Visualization

If you want a quick lightcurve visualization of an object without going to the Fink portal, run `mcvs.utils.fink_lightcurve('ZTF_ID')`

## 📧 Contact

If you have any question about some functionality of the package, do not hesitate to contact me at clems.mur@gmail.com.