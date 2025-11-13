import functools
import logging
import sys

LOGGER_NAME = "custom_ocr"
_logger = logging.getLogger(LOGGER_NAME)


def debug(msg):
    _logger.debug(msg)


def info(msg):
    _logger.info(msg)


def warning(msg):
    _logger.warning(msg)


@functools.lru_cache(None)
def warning_once(msg):
    """
    This method is identical to `logger.warning()`, but will emit the warning with the same message only once
    """
    warning(msg)


def error(msg):
    _logger.error(msg)


def critical(msg):
    _logger.critical(msg)


def exception(msg):
    _logger.exception(msg)


def setup_logging(verbosity: str):
    """setup logging level

    Args:
        verbosity (str, optional): the logging level, `DEBUG`, `INFO`, `WARNING`. Defaults to None.
    """
    _configure_logger(_logger, verbosity.upper())


def _configure_logger(logger, verbosity):
    """_configure_logger"""
    if verbosity == "DEBUG":
        _logger.setLevel(logging.DEBUG)
    elif verbosity == "INFO":
        _logger.setLevel(logging.INFO)
    elif verbosity == "WARNING":
        _logger.setLevel(logging.WARNING)
    logger.propagate = False
    if not logger.hasHandlers():
        _add_handler(logger)


def _add_handler(logger):
    """_add_handler"""
    logfmt = "%(asctime)s %(levelname)-8s pid:%(process)d %(name)s:%(lineno)03d:%(funcName)s %(message)s"
    logfmt = logging.Formatter(logfmt)
    handler = logging.StreamHandler(sys.stdout)
    # handler.setFormatter(logfmt)
    logger.addHandler(handler)
