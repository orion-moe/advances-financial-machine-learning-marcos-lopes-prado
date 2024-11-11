import logging
from config.config import LOG_FILE

def setup_logging():
    logging.basicConfig(
        filename=LOG_FILE,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
