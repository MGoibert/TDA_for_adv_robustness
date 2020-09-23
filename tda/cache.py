import inspect
import os
import torch
import pathlib
import socket

from tda.rootpath import rootpath
from tda.tda_logging import get_logger

logger = get_logger("Cache")


if os.path.exists("/var/opt/data/user_data"):
    # We are on gpu
    cache_root = f"/var/opt/data/user_data/tda/"
elif "mesos" in socket.gethostname():
    # We are in mozart
    cache_root = f"{os.environ['HOME']}/tda_cache/"
else:
    # Other cases (local)
    cache_root = f"{rootpath}/cache/"

logger.info(f"Cache root {cache_root}")


def cached(my_func):
    arg_spec = inspect.getfullargspec(my_func).args

    def my_func_with_cache(*args, **kw):
        kw.update({arg_spec[i]: arg for i, arg in enumerate(args)})
        base_path = f"{cache_root}/{my_func.__name__}/"
        pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)
        cache_path = (
            base_path
            + "_".join(sorted([f"{k}={str(v)}" for (k, v) in kw.items()]))
            + ".cached"
        )

        if os.path.exists(cache_path):
            logger.info(
                f"Using cache file {cache_path} for the call to {my_func.__name__}"
            )
            return torch.load(cache_path)
        else:
            logger.info(
                f"No cache found in {cache_path} for the call to {my_func.__name__}"
            )
            ret = my_func(**kw)
            logger.info(
                f"Creating cache file {cache_path} for the call to {my_func.__name__}"
            )
            torch.save(ret, cache_path, pickle_protocol=4)
            return ret

    return my_func_with_cache
