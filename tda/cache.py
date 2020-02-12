import inspect
import os
import torch
import pathlib
import tensorflow as tf
import pickle

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
            torch.save(ret, cache_path, pickle_protocol=4)
            return ret

    return my_func_with_cache


def hdfs_cached(my_func):
    arg_spec = inspect.getfullargspec(my_func).args

    def my_func_with_cache(*args, **kw):
        kw.update({
            arg_spec[i]: arg for i, arg in enumerate(args)
        })
        base_path = f"hdfs://root/user/{os.environ['CRITEO_USER']}/tda_cache/{my_func.__name__}/"
        cache_path = base_path + "_".join(sorted([f"{k}={str(v)}" for (k, v) in kw.items()])) + ".cached"

        if tf.io.gfile.exists(cache_path):
            logger.info(f"Using cache file {cache_path} for the call to {my_func.__name__}")
            with tf.io.gfile.GFile(cache_path, "rb") as hdfs_file:
                return pickle.load(hdfs_file)
        else:
            ret = my_func(**kw)
            logger.info(f"Creating cache file {cache_path} for the call to {my_func.__name__}")
            with tf.io.gfile.GFile(cache_path, "wb") as hdfs_file:
                pickle.dump(ret, hdfs_file)
            return ret

    return my_func_with_cache
