import json
import redis
import pandas as pd

from config.development import REDIS_STREAM_UP
from config.settings import REDIS_CHANNEL_FEEDBACK


class RedisClient:

    def __init__(self, url: str = 'redis://127.0.0.1:6379/0'):
        self.redis = redis.Redis.from_url(url=url)

    def get(self, key):
        return self.redis.get(key)

    def set(self, key, value):
        self.redis.set(key, value)

    def delete(self, key):
        self.redis.delete(key)

    def keys(self, pattern='*'):
        return self.redis.keys(pattern)

    # set dataframe to redis
    def pdset(self, key, df):
        self.redis.set(key, df.to_msgpack(compress='zlib'))

    # get dataframe from redis
    def pdget(self, key):
        return pd.read_msgpack(self.redis.get(key))

    # feedback to redis channel
    def feedback(self, json_data):
        self.redis.publish(REDIS_CHANNEL_FEEDBACK, json.dumps(json_data))

    def notify(self, json_data):
        self.redis.xadd(REDIS_STREAM_UP, {'msg': json.dumps(json_data)})
