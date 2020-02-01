from tda.cache import cached
from tda.logging import get_logger

logger = get_logger("TestCache")


def test_cached():

    @cached
    def f(x, y):
        return 2 * x + y

    logger.info(f(1, y=2))
    logger.info(f(2, 3))
    logger.info(f(1, 2))