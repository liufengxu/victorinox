import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler

__all__ = ["LOGGER", "init_logger"]
logging.basicConfig(stream=sys.stdout,
                    filemode='a',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    format='%(asctime)s.%(msecs)03d %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')

LOGGER = logging.getLogger('default')


def init_logger(name=None):
    pattern = "logs/{0}.log"
    default_name = "exposure_{0}".format(os.getpid())
    name = default_name if not name else name
    logFormater = logging.Formatter("%(asctime)s.%(msecs)03d %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
                                    datefmt='%Y-%m-%d %H:%M:%S')
    fileHandler = TimedRotatingFileHandler(pattern.format(name), when="D", backupCount=10)
    fileHandler.setFormatter(logFormater)
    LOGGER.addHandler(fileHandler)

