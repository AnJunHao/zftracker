
TQDM = None

def set_global_tqdm(notebook=True, disable=False):
    global TQDM
    if notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    TQDM = tqdm
    if disable:
        TQDM = lambda x, *args, **kwargs: x

set_global_tqdm()