import os

from torchrl.record.loggers import generate_exp_name

from causal_meta.record.logger import get_logger


def build_logger(cfg, name="mpc", log_dir=""):
    exp_name = generate_exp_name(name.upper(), cfg.exp_name)
    if log_dir == "":
        log_dir = os.path.join(os.getcwd(), name)

    logging_cfg = dict(cfg)
    logging_cfg.update({"log_dir": log_dir})

    if cfg.logger == 'wandb':
        wandb_kwargs = {
            "project": "causal_meta",
            "entity": "causal_focus",
            "group": f"{name.upper()}_{cfg.env_name}",
            "config": logging_cfg,
            "offline": cfg.offline_logging,
        }
    else:
        wandb_kwargs = None

    logger = get_logger(
        logger_type=cfg.logger,
        logger_name=log_dir,
        experiment_name=exp_name,
        wandb_kwargs=wandb_kwargs
    )
    return logger
