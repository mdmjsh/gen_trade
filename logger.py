import logging
import sys


def get_logger(name):
    logging.basicConfig(
        format="%(asctime)s%(levelname)s:%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )
    return logging.getLogger(name)
