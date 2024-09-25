import logging
import pathlib


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    # for level in logging_levels:
    #     setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


# https://github.com/Borda/BIRL/blob/087e6771236ebffc4d293e37076815438d88e6c2/birl/utilities/experiments.py#L262-L273
def release_logger_files():
    """close all handlers to a file
    >>> release_logger_files()
    >>> len([1 for lh in logging.getLogger().handlers
    ...      if type(lh) is logging.FileHandler])
    0
    """
    for hl in logging.getLogger().handlers:
        if isinstance(hl, logging.FileHandler):
            hl.close()
            logging.getLogger().removeHandler(hl)


# https://github.com/Borda/BIRL/blob/087e6771236ebffc4d293e37076815438d88e6c2/birl/utilities/experiments.py#L276-L298
#: default logging template - log location/source for logging to file
STR_LOG_FORMAT = "[%(asctime)s]:[%(levelname)s]@[%(filename)s]:[%(name)s]:[%(processName)s]:[%(threadName)s]:[pluplu]- %(message)s"
#: default logging template - date-time for logging to file
LOG_FILE_FORMAT = logging.Formatter(STR_LOG_FORMAT, datefmt="%H:%M:%S")


def set_experiment_logger(
    path_out: str, file_name: str = "out.txt", reset: bool = True
) -> None:
    """set the logger to file
    :param str path_out: path to the output folder
    :param str file_name: log file name
    :param bool reset: reset all previous logging into a file
    >>> set_experiment_logger('.')
    >>> len([1 for lh in logging.getLogger().handlers
    ...      if type(lh) is logging.FileHandler])
    1
    >>> release_logger_files()
    >>> os.remove(FILE_LOGS)
    """
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    if reset:
        release_logger_files()
    path_logger = pathlib.Path(path_out) / file_name
    fh = logging.FileHandler(path_logger)
    fh.setLevel(logging.INFO)
    fh.setFormatter(LOG_FILE_FORMAT)
    log.addHandler(fh)

    logging.captureWarnings(capture=True)

    return log
