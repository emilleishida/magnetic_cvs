from tqdm import tqdm

# Wrapper for tqdm with default arguments:
def tqdm2(iterable, **kwargs):
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