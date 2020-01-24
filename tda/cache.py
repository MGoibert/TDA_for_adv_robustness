import inspect
import os
import torch
import pathlib

from tda.rootpath import rootpath
from tda.logging import get_logger

logger = get_logger("Cache")


def cached(my_func):
    arg_spec = inspect.getfullargspec(my_func).args

    def my_func_with_cache(*args, **kw):
        kw.update({
            arg_spec[i]: arg for i, arg in enumerate(args)
        })
        base_path = f"{rootpath}/cache/{my_func.__name__}/"
        pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)
        cache_path = base_path + "_".join(sorted([f"{k}={str(v)}" for (k, v) in kw.items()])) + ".cached"

        if os.path.exists(cache_path):
            logger.info(f"Using cache file {cache_path} for the call to {my_func.__name__}")
            return torch.load(cache_path)
        else:
            ret = my_func(**kw)
            logger.info(f"Creating cache file {cache_path} for the call to {my_func.__name__}")
            torch.save(ret, cache_path)
            return ret

    return my_func_with_cache
