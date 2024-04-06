import os
import tempfile

from torchrl.record.loggers import CSVLogger, TensorboardLogger

from intact.record.logger import MultipleLoggerWrapper, MeanScalarWrapper

tmp_dir = tempfile.gettempdir()


def test_multiple_logger():
    log_dir = os.path.join(tmp_dir, "test_multiple_logger")
    logger = MultipleLoggerWrapper(
        [CSVLogger("test", log_dir), TensorboardLogger("test", log_dir)]
    )

    logger.log_scalar("t1", 1)


def test_mean_scalar_wrapper():
    log_dir = os.path.join(tmp_dir, "test_multiple_logger")

    logger = CSVLogger("test", log_dir)
    logger = MeanScalarWrapper(logger)

    logger.add_scaler("t1", 1)
    logger.add_scaler("t1", 2)
    logger.dump_scaler(1)

    logger.add_scaler("t1", 3)
    logger.dump_scaler(2)
