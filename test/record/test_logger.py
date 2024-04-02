from causal_meta.record.logger import CSVLogger, TensorboardLogger, MultipleLoggerWrapper, MeanScalarWrapper


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
