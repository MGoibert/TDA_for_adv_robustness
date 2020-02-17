import logging
import os

logging.basicConfig(level=logging.INFO)
root_logger = logging.getLogger()
root_logger.handlers = list()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    my_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    my_handler.setFormatter(formatter)
    logger.handlers = [my_handler]

    if "TDA_LOG_PATH" in os.environ:
        log_path = os.environ["TDA_LOG_PATH"]
        file_handler = logging.FileHandler(log_path)
        logger.handlers.append(file_handler)

    return logger
