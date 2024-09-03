import logging
import os
import sys
from datetime import datetime

# Log file format
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Here we get the path to store the logs file
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# This makes sure the logs folder is not create if it all ready exist
os.makedirs(logs_path, exist_ok=True)


# Complete log path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)


# Configuration setting for logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
