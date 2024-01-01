from typing import Dict, Sequence, Union
from collections import defaultdict

from torch import Tensor
from torchrl.record.loggers import Logger, CSVLogger, TensorboardLogger, WandbLogger
from torchrl.record.loggers import get_logger as _get_logger


class MultipleLoggerWrapper(Logger):
    def __init__(self, loggers):
        assert len(loggers) > 0
        super().__init__(exp_name=loggers[0].exp_name, log_dir=loggers[0].log_dir)

        self.loggers = loggers

    def _create_experiment(self) -> "Experiment":  # noqa: F821
        pass

    def log_scalar(self, name: str, value: float, step: int = None) -> None:
        for logger in self.loggers:
            logger.log_scalar(name=name, value=value, step=step)

    def log_video(self, name: str, video: Tensor, step: int = None, **kwargs) -> None:
        for logger in self.loggers:
            logger.log_video(name=name, video=video, step=step)

    def log_hparams(self, cfg: Union["DictConfig", Dict]) -> None:  # noqa: F821
        for logger in self.loggers:
            logger.log_hparams(cfg=cfg)

    def __repr__(self) -> str:
        loggers_repr = "\n".join([repr(logger) for logger in self.loggers])
        return f"MultipleLogger({loggers_repr})"

    def log_histogram(self, name: str, data: Sequence, **kwargs):
        for logger in self.loggers:
            logger.log_histogram(name=name, data=data)


class AverageMeter:
    def __init__(self):
        self._sum = 0.0
        self._count = 0

    def update(self, value: float, n: int = 1):
        if isinstance(value, Tensor):
            value = value.detach().item()
        self._sum += value
        self._count += n

    def value(self) -> float:
        return self._sum / max(1, self._count)


class MeanScalarWrapper(Logger):
    def __init__(self, logger: Logger):
        super().__init__(exp_name=logger.exp_name, log_dir=logger.log_dir)

        self._cache = defaultdict(AverageMeter)
        self.logger = logger

    def _create_experiment(self) -> "Experiment":  # noqa: F821
        pass

    def log_scalar(self, name: str, value: float, step: int = None) -> None:
        self.logger.log_scalar(name, value, step)

    def log_video(self, name: str, video: Tensor, step: int = None, **kwargs) -> None:
        self.logger.log_video(name, video, step, **kwargs)

    def log_hparams(self, cfg: Union["DictConfig", Dict]) -> None:  # noqa: F821
        self.logger.log_hparams(cfg)

    def __repr__(self) -> str:
        return f"MeanScalarWrapper({repr(self.logger)})"

    def log_histogram(self, name: str, data: Sequence, **kwargs):
        self.logger.log_histogram(name, data)

    def add_scaler(self, name, value):
        self._cache[name].update(value)

    def dump_scaler(self, step):
        for name in self._cache:
            self.log_scalar(name, self._cache[name].value(), step)
        self._cache = defaultdict(AverageMeter)


def get_logger(
        logger_type: Union[str, list], logger_name: str, experiment_name: str, mean_scaler=True, **kwargs
) -> Logger:
    if isinstance(logger_type, str):
        logger = _get_logger(logger_type, logger_name, experiment_name, **kwargs)
    else:
        loggers = [_get_logger(t, logger_name, experiment_name, **kwargs) for t in logger_type]
        logger = MultipleLoggerWrapper(loggers)

    if mean_scaler:
        logger = MeanScalarWrapper(logger)
    return logger


def test_multiple_logger():
    logger = MultipleLoggerWrapper([CSVLogger("1", "1"), TensorboardLogger("1", "1")])

    logger.log_scalar("1", 1)


def test_mean_scalar_wrapper():
    logger = CSVLogger("1", "1")
    logger = MeanScalarWrapper(logger)

    logger.add_scaler("1", 1)
    logger.add_scaler("1", 2)
    logger.dump_scaler(1)

    logger.add_scaler("1", 3)
    logger.dump_scaler(2)


if __name__ == '__main__':
    test_mean_scalar_wrapper()
