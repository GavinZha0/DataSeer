import os
import time
from loguru import logger
from config.settings import BASE_DIR


# remove console output
logger.remove(handler_id=None)

log_path = os.path.join(BASE_DIR, 'logs')
if not os.path.exists(log_path):
    os.mkdir(log_path)

log_path_info = os.path.join(log_path, f'info_{time.strftime("%Y-%m-%d")}.log')
log_path_error = os.path.join(log_path, f'error_{time.strftime("%Y-%m-%d")}.log')

info = logger.add(log_path_info, rotation="00:00", retention="3 days", enqueue=True, encoding="UTF-8", level="INFO")
error = logger.add(log_path_error, rotation="00:00", retention="3 days", enqueue=True, encoding="UTF-8", level="ERROR")
