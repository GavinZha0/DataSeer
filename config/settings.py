# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2021/10/19
# @File           : settings.py
# @desc           : settings

import os
from fastapi.security import OAuth2PasswordBearer

"""
Project info
"""
TITLE = "DataSeer"
VERSION = "0.1.0"
DESCRIPTION = "Python server for AI and BI!"

"""
env config. development has more debug info
"""
DEVELOPMENT = True

if DEVELOPMENT:
    from config.development import *
else:
    from config.production import *

"""
Root dir
"""
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

"""
Auth config
"""
OAUTH_ENABLE = True
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False) if OAUTH_ENABLE else lambda: ""
"""JWT secret key"""
SECRET_KEY = 'Good good study! Day day up! Then walk out to have a look at the beautiful world with you family!'
"""JWT algorithm"""
ALGORITHM = "HS256"
"""token expire time (min)"""
ACCESS_TOKEN_EXPIRE_MINUTES = 60
"""token refresh time"""
REFRESH_TOKEN_EXPIRE_MINUTES = 1440 * 2
"""token cache time"""
ACCESS_TOKEN_CACHE_MINUTES = 30
ACCESS_TOKEN_FIELD = "access-token"
SHADOW_TOKEN_FIELD = "shadow-token"

"""
static folder
"""
STATIC_ENABLE = True
STATIC_URL = "/media"
STATIC_DIR = "static"
STATIC_ROOT = os.path.join(BASE_DIR, STATIC_DIR)
TEMP_DIR = os.path.join(BASE_DIR, "temp")


"""
cross solution
"""
CORS_ORIGIN_ENABLE = True
ALLOW_ORIGINS = ["*"]
# allow cookie
ALLOW_CREDENTIALS = True
# all method, get, post, put...
ALLOW_METHODS = ["*"]
# allow header
ALLOW_HEADERS = ["*"]

"""
global events
"""
EVENTS = [
    "core.event.connect_mongo" if MONGO_DB_ENABLE else None,
    "core.event.connect_redis" if REDIS_ENABLE else None,
]

"""
Others
"""
# login log
LOGIN_LOG_RECORD = True
# local log
REQUEST_LOG_RECORD = False
# log to mongoDb
OPERATION_LOG_RECORD = False
# log filter
OPERATION_RECORD_METHOD = ["POST", "PUT", "DELETE"]
# ignore operation
IGNORE_OPERATION_FUNCTION = ["post_dicts_details"]
# zip http response
HTTP_RESPONSE_ZIP = False
"""
middle wares
"""
MIDDLEWARES = [
    "core.middleware.register_request_log_middleware" if REQUEST_LOG_RECORD else None,
    "core.middleware.register_operation_record_middleware" if OPERATION_LOG_RECORD and MONGO_DB_ENABLE else None,
    "core.middleware.register_jwt_refresh_middleware"
]

"""
Ray config
"""
RAY_ENABLE = False
RAY_LOCAL_MODE = True
RAY_NUM_CPU = 1
RAY_NUM_GPU = 0

"""
FTP server config
"""
FTP_SERVER_PATH = "C:/Users/jichu/Workspace/ftp/"