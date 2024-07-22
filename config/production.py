# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2021/10/19
# @File           : production.py
# @desc           : production config


"""
Database config
"""
# main database of DataPie
SQLALCHEMY_DATABASE_URL = "mysql+asyncmy://admin:admin520@datapie.cnqbtlcpe5hy.us-east-2.rds.amazonaws.com:3306/datapie"
# mlflow db to track ML lifecycle
SQLALCHEMY_MLFLOW_DB_URL = "mysql+pymysql://admin:admin520@datapie.cnqbtlcpe5hy.us-east-2.rds.amazonaws.com:3306/mlflow"

"""
Redis config
"""
REDIS_ENABLE = True
REDIS_URL = "redis://127.0.0.1:6379/1"
REDIS_HOST = '127.0.0.1'
REDIS_PORT = 6379
REDIS_DB = 0
# used to receive task
REDIS_STREAM_NAME = "HUDSON"
# local consumer group
REDIS_CONSUMER_GROUP = "NYC"
# unique consumer name
REDIS_CONSUMER_NAME = "nyc_01"
# used to report task status and result
REDIS_CHANNEL_FEEDBACK = 'feedback'
# used for timer task
REDIS_CHANNEL_APSCHEDULER = "apscheduler"

"""
MongoDb config
"""
MONGO_DB_ENABLE = False
MONGO_DB_NAME = "datapie"
MONGO_DB_URL = f"mongodb://username:password@127.0.0.1:27017/?authSource={MONGO_DB_NAME}"

"""
AWS S3 config
"""
AWS_S3_ENDPOINT = "https://s3.us-east-2.amazonaws.com"
AWS_S3_ACCESS_KEY = "AKIAYSIQQG2SQMGUGNJK"
AWS_S3_SECRET_KEY = "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"
AWS_S3_BUCKET = "datapie"
