import pandas as pd
from pathlib import Path


def get_mCVs_path() -> Path:
    """
    Get the absolute path to the mCVs CSV file.
    """
    return str(Path(__file__).parent / 'data' / 'mCVs.csv')


def add_new_mCVs(objectIds: str | list[str],
                 is_polar: bool | list[bool],
                 is_bogus: bool | list[bool] = False,
                 objectId2: str | list[str | None] | None = None
                 ) -> None:
    """
    Add new mCVs to the current mCVs list.
    
    Parameters
    ---
        objectIds: str | list[str]
            ObjectId or list of objectIds to be added to the current mCVs list.
        
        is_polar: bool | list[bool]
            Wether to tag added objects as polars (True) or not (False).
        
        is_bogus: bool | list[bool], optional
            Wether to tag added objects as bogus (True) or not (False). Defaults to False.
        
        objectId2: str | list[str | None] | None, optional
            Object's second identifier, if applicable. Lightcurves from the two Ids will be treated as one. If provided, should match the length of `objectIds`. Defaults to None.
    """
    
    mCVs = pd.read_csv(get_mCVs_path())

    if isinstance(objectIds, str):
        objectIds = [objectIds]

    # Check if user tries to add already existing mCVs:
    for obj in objectIds:
        if obj in mCVs['objectId'].values:
            raise ValueError(f'{obj} already exists in current mCVs list.')
    
    # Adding new mCVs:
    new_mCVs = pd.DataFrame({'objectId': objectIds, 'is_polar': is_polar, 'is_bogus': is_bogus, 'objectId2': objectId2 if objectId2 is not None else [None] * len(objectIds)})
    mCVs = pd.concat([mCVs, new_mCVs], ignore_index=True)
    
    # Save updated mCVs DataFrame:
    mCVs.to_csv(get_mCVs_path(), index=False)
    
    print(f'{", ".join(objectIds)} successfully added to mCVs list.')


def remove_mCVs(objectIds: str | list[str]) -> None:
    """
    Remove mCVs from the current mCVs list.
    
    Parameters
    ---
        objectIds: str | list[str]
            ObjectId or list of objectIds to be removed from the current mCVs list.
    """
    
    mCVs = pd.read_csv(get_mCVs_path())

    if isinstance(objectIds, str):
        objectIds = [objectIds]
    
    # Check if user tries to remove non-existing mCVs:
    for obj in objectIds:
        if obj not in mCVs['objectId'].values:
            raise ValueError(f'{obj} not found in current mCVs list. Please update the list with `add_new_mCVs` function.')

    # Ask for confirmation:
    confirmation = input(f'Please confirm removal of {len(objectIds)} mCV(s): {", ".join(objectIds)}. [Y/n]: ')
    if confirmation != 'Y':
        print('Removal cancelled.')
        return

    # Removing mCVs:
    mCVs = mCVs[~mCVs['objectId'].isin(objectIds)]

    # Save updated mCVs DataFrame:
    mCVs.to_csv(get_mCVs_path(), index=False)
    
    print(f'{", ".join(objectIds)} successfully removed from mCVs list.')


def tag_polars(objectIds: str | list[str],
               not_polar: bool = False
               ) -> None:
    """
    Tag given objectId(s) as polar(s).
    
    Parameters
    ---
        objectId: str | list[str]
            The objectId or list of objectIds to be tagged as polar(s).
        
        not_polar: bool, optional
            If True, tags the objectId(s) as non-polar(s) instead. Defaults to False.
    """
    
    mCVs = pd.read_csv(get_mCVs_path())
    
    if isinstance(objectIds, str):
        objectIds = [objectIds]
    
    # Check if user tries to tag non-existing mCVs:
    for obj in objectIds:
        if obj not in mCVs['objectId'].values:
            raise ValueError(f'{obj} not found in current mCVs list. Please update the list with `add_new_mCVs` function.')

    # Tagging mCVs:
    if not_polar:
        mCVs.loc[mCVs['objectId'].isin(objectIds), 'is_polar'] = False
    else:
        mCVs.loc[mCVs['objectId'].isin(objectIds), 'is_polar'] = True
    
    # Save updated mCVs DataFrame:
    mCVs.to_csv(get_mCVs_path(), index=False)
    
    print(f'{", ".join(objectIds)} successfully tagged as {"non-polar" if not_polar else "polar"}.')


def tag_bogus(objectIds: str | list[str],
              not_bogus: bool = False
              ) -> None:
    """
    Tag given objectId(s) as bogus.
    
    Parameters
    ---
        objectId: str | list[str]
            The objectId or list of objectIds to be tagged as bogus.
        
        not_bogus: bool, optional
            If True, tags the objectId(s) as non-bogus instead. Defaults to False.
    """
    
    mCVs = pd.read_csv(get_mCVs_path())
    
    if isinstance(objectIds, str):
        objectIds = [objectIds]
    
    # Check if user tries to tag non-existing mCVs:
    for obj in objectIds:
        if obj not in mCVs['objectId'].values:
            raise ValueError(f'{obj} not found in current mCVs list. Please update the list with `add_new_mCVs` function.')

    # Tagging mCVs:
    if not_bogus:
        mCVs.loc[mCVs['objectId'].isin(objectIds), 'is_bogus'] = False
    else:
        mCVs.loc[mCVs['objectId'].isin(objectIds), 'is_bogus'] = True
    
    # Save updated mCVs DataFrame:
    mCVs.to_csv(get_mCVs_path(), index=False)
    
    print(f'{", ".join(objectIds)} successfully tagged as {"non-bogus" if not_bogus else "bogus"}.')


