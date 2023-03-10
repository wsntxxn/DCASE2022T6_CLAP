import logging
import logging.config
from pathlib import Path
from utils.util import read_json

def setup_logging(save_dir, filename=None, log_config='logger/logger_config.json',
    default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                if filename:
                    handler['filename'] = str(save_dir / filename)
                else:
                    handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)

    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
    
    return config["handlers"]["info_file_handler"]["filename"]
