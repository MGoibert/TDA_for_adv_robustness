import logging

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
    return logger
