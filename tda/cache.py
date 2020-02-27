import inspect
import os
import torch
import pathlib
import pickle

from tda.rootpath import rootpath
from tda.tda_logging import get_logger

logger = get_logger("Cache")

if os.path.exists("/var/opt/data/user_data"):
    on_gpu = True
else:
    on_gpu = False

logger.info(f"On Gpu ? {on_gpu}")

def cached(my_func):
    arg_spec = inspect.getfullargspec(my_func).args

    def my_func_with_cache(*args, **kw):
        kw.update({
            arg_spec[i]: arg for i, arg in enumerate(args)
        })
        if not on_gpu:
            base_path = f"{rootpath}/cache/{my_func.__name__}/"
        else:
            base_path = f"/var/opt/data/user_data/tda/{my_func.__name__}/"
        pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)
        cache_path = base_path + "_".join(sorted([f"{k}={str(v)}" for (k, v) in kw.items()])) + ".cached"

        if os.path.exists(cache_path):
            logger.info(f"Using cache file {cache_path} for the call to {my_func.__name__}")
            return torch.load(cache_path)
        else:
            ret = my_func(**kw)
            logger.info(f"Creating cache file {cache_path} for the call to {my_func.__name__}")
            torch.save(ret, cache_path, pickle_protocol=4)
            return ret

    return my_func_with_cache
