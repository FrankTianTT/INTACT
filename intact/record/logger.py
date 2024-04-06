from collections import defaultdict
from typing import Dict, Sequence, Union

from torch import Tensor
from torchrl.record.loggers import Logger
from torchrl.record.loggers import get_logger as _get_logger


class MultipleLoggerWrapper(Logger):
    """
    A wrapper class for multiple loggers.

    This class allows you to use multiple loggers as if they were a single logger.
    """

    def __init__(self, loggers):
        """
        Initialize the MultipleLoggerWrapper.

        Args:
            loggers (list): A list of Logger instances.
        """
        assert len(loggers) > 0
        super().__init__(
            exp_name=loggers[0].exp_name, log_dir=loggers[0].log_dir
        )

        self.loggers = loggers

    def _create_experiment(self) -> "Experiment":  # noqa: F821
        pass

    def log_scalar(self, name: str, value: float, step: int = None) -> None:
        """
        Log a scalar value.

        Args:
            name (str): The name of the scalar.
            value (float): The value of the scalar.
            step (int, optional): The current step. Defaults to None.
        """
        for logger in self.loggers:
            logger.log_scalar(name=name, value=value, step=step)

    def log_video(
        self, name: str, video: Tensor, step: int = None, **kwargs
    ) -> None:
        """
        Log a video.

        Args:
            name (str): The name of the video.
            video (Tensor): The video to log.
            step (int, optional): The current step. Defaults to None.
        """
        for logger in self.loggers:
            logger.log_video(name=name, video=video, step=step)

    def log_hparams(
        self, cfg: Union["DictConfig", Dict]
    ) -> None:  # noqa: F821
        """
        Log hyperparameters.

        Args:
            cfg (Union["DictConfig", Dict]): The hyperparameters to log.
        """
        for logger in self.loggers:
            logger.log_hparams(cfg=cfg)

    def __repr__(self) -> str:
        """
        Return a string representation of the MultipleLoggerWrapper.

        Returns:
            str: A string representation of the MultipleLoggerWrapper.
        """
        loggers_repr = "\n".join([repr(logger) for logger in self.loggers])
        return f"MultipleLogger({loggers_repr})"

    def log_histogram(self, name: str, data: Sequence, **kwargs):
        """
        Log a histogram.

        Args:
            name (str): The name of the histogram.
            data (Sequence): The data for the histogram.
        """
        for logger in self.loggers:
            logger.log_histogram(name=name, data=data)


class AverageMeter:
    """
    A class that maintains the sum and count of values to compute an average.

    This class is useful for tracking averages over a series of values.
    """

    def __init__(self):
        """
        Initialize the AverageMeter.

        The sum and count are initialized to 0.
        """
        self._sum = 0.0
        self._count = 0

    def update(self, value: float, n: int = 1):
        """
        Update the sum and count with a new value.

        Args:
            value (float): The new value to add to the sum.
            n (int, optional): The number of instances of the value. Defaults to 1.
        """
        if isinstance(value, Tensor):
            value = value.detach().item()
        self._sum += value
        self._count += n

    def value(self) -> float:
        """
        Compute the average of the values seen so far.

        Returns:
            float: The average of the values.
        """
        return self._sum / max(1, self._count)


class MeanScalarWrapper(Logger):
    """
    A wrapper class for a Logger that computes the mean of scalar values.

    This class extends the Logger class and adds functionality for computing the mean of scalar values.
    It maintains a cache of AverageMeter instances for each scalar, which are used to compute the mean.
    """

    def __init__(self, logger: Logger):
        """
        Initialize the MeanScalarWrapper.

        Args:
            logger (Logger): The Logger instance to wrap.
        """
        super().__init__(exp_name=logger.exp_name, log_dir=logger.log_dir)

        self._cache = defaultdict(AverageMeter)
        self.logger = logger

    def _create_experiment(self) -> "Experiment":  # noqa: F821
        pass

    def log_scalar(self, name: str, value: float, step: int = None) -> None:
        """
        Log a scalar value.

        Args:
            name (str): The name of the scalar.
            value (float): The value of the scalar.
            step (int, optional): The current step. Defaults to None.
        """
        self.logger.log_scalar(name, value, step)

    def log_video(
        self, name: str, video: Tensor, step: int = None, **kwargs
    ) -> None:
        """
        Log a video.

        Args:
            name (str): The name of the video.
            video (Tensor): The video to log.
            step (int, optional): The current step. Defaults to None.
        """
        self.logger.log_video(name, video, step, **kwargs)

    def log_hparams(
        self, cfg: Union["DictConfig", Dict]
    ) -> None:  # noqa: F821
        """
        Log hyperparameters.

        Args:
            cfg (Union["DictConfig", Dict]): The hyperparameters to log.
        """
        self.logger.log_hparams(cfg)

    def __repr__(self) -> str:
        """
        Return a string representation of the MeanScalarWrapper.

        Returns:
            str: A string representation of the MeanScalarWrapper.
        """
        return f"MeanScalarWrapper({repr(self.logger)})"

    def log_histogram(self, name: str, data: Sequence, **kwargs):
        """
        Log a histogram.

        Args:
            name (str): The name of the histogram.
            data (Sequence): The data for the histogram.
        """
        self.logger.log_histogram(name, data)

    def add_scaler(self, name, value):
        """
        Add a scalar value to the cache.

        Args:
            name (str): The name of the scalar.
            value (float): The value of the scalar.
        """
        self._cache[name].update(value)

    def dump_scaler(self, step):
        """
        Log the mean of the scalar values in the cache and clear the cache.

        Args:
            step (int): The current step.
        """
        for name in self._cache:
            self.log_scalar(name, self._cache[name].value(), step)
        self._cache = defaultdict(AverageMeter)


def get_logger(
    logger_type: Union[str, list],
    logger_name: str,
    experiment_name: str,
    mean_scaler=True,
    **kwargs,
) -> Logger:
    if isinstance(logger_type, str):
        logger = _get_logger(
            logger_type, logger_name, experiment_name, **kwargs
        )
    else:
        loggers = [
            _get_logger(t, logger_name, experiment_name, **kwargs)
            for t in logger_type
        ]
        logger = MultipleLoggerWrapper(loggers)

    if mean_scaler:
        logger = MeanScalarWrapper(logger)
    return logger
