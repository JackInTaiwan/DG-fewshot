import os
import logging
import logging.config



def logging_config(save_dir=None, log_file_path=None):
    """Set up logging config for training."""

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "[%(asctime)s][%(name)s][%(funcName)s][%(levelname)s] %(message)s"
            },
            "simple": {
                "format": "[%(asctime)s][%(levelname)s] %(message)s"
            },
        },
        "handlers": {
            "terminal": {
                "level": "INFO",
                "formatter": "simple",
                "class": "logging.StreamHandler",
            }
        },
        "loggers": {
        },
        "root": {
            "handlers": ["terminal"],
            "level": "INFO",
        }
    }

    if save_dir or log_file_path:
        log_file_path = log_file_path or os.path.join(save_dir, "training.log")

        if not os.path.exists(os.path.dirname(log_file_path)):
            os.makedirs(os.path.dirname(log_file_path))

        config["handlers"]["file"] = {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.FileHandler",
                "filename": log_file_path,
                "mode": "a+",
        }
        config["root"]["handlers"].append("file")

    logging.config.dictConfig(config)

