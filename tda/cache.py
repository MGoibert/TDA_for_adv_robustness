import inspect
import os
import torch
import pathlib
import tensorflow as tf
import pickle

from tda.rootpath import rootpath
from tda.tda_logging import get_logger

logger = get_logger("Cache")

try:
    tf.io.gfile.listdir(f"hdfs://root/user/{os.environ['USER']}")
    hdfs_available = True
except Exception as e:
    hdfs_available = False

if os.path.exists("/var/opt/data/user_data"):
    on_gpu = True
else:
    on_gpu = False


logger.info(f"Hdfs is available ? {hdfs_available}")
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


def hdfs_cached(my_func):
    arg_spec = inspect.getfullargspec(my_func).args

    if not hdfs_available:
        return my_func

    def my_func_with_cache(*args, **kw):
        kw.update({
            arg_spec[i]: arg for i, arg in enumerate(args)
        })
        base_path = f"hdfs://root/user/{os.environ['CRITEO_USER']}/tda_cache/{my_func.__name__}/"
        cache_path = base_path + "_".join(sorted([f"{k}={str(v)}" for (k, v) in kw.items()])) + ".cached"

        if tf.io.gfile.exists(cache_path):
            try:
                logger.info(f"Using cache file {cache_path} for the call to {my_func.__name__}")
                with tf.io.gfile.GFile(cache_path, "rb") as hdfs_file:
                    return pickle.load(hdfs_file)
            except Exception as e:
                logger.warn(f"Error while loading cache file... ({e})")

        # Cleaning if needed
        try:
            tf.io.gfile.remove(cache_path)
        except Exception:
            pass

        ret = my_func(**kw)
        logger.info(f"Creating cache file {cache_path} for the call to {my_func.__name__}")

        try:
            with tf.io.gfile.GFile(cache_path, "wb") as hdfs_file:
                pickle.dump(ret, hdfs_file, protocol=4)
        except Exception as e:
            logger.warn(f"Unable to save file to path {cache_path} ({e})")

        return ret

    return my_func_with_cache
