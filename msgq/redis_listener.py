import time
from threading import Thread
import redis
from config.settings import REDIS_STREAM_DOWN, REDIS_CONSUMER_GROUP, REDIS_CONSUMER_NAME, REDIS_CHANNEL_FEEDBACK
import json

class RedisListener(object):
    def __init__(self, url: str = 'redis://127.0.0.1:6379/0'):
        self.redis = redis.Redis.from_url(url=url)

        try:
            # get stream info
            sinfo = self.redis.xinfo_stream(REDIS_STREAM_DOWN)
            # print(sinfo)
        except Exception:
            # create stream if it doesn't exist
            self.redis.xgroup_create(REDIS_STREAM_DOWN, REDIS_CONSUMER_GROUP, id='$', mkstream=True)
            pass

        self.thread = Thread(target=self.loop_consuming, daemon=True)
        self.thread.start()

    def loop_consuming(self):
        while True:
            self.consume(REDIS_STREAM_DOWN, REDIS_CONSUMER_GROUP, REDIS_CONSUMER_NAME, target=self.msg_handler)

    def consume(self, stream_name, consumer_group, consumer_name, target=None):
        message = self.redis.xreadgroup(consumer_group, consumer_name, {stream_name: '>'}, block=0, count=1)
        if not message or not message[0] or not message[0][1]:
           return

        msg_list = message[0][1]
        for msg in msg_list:
           try:
               msg_id, data = msg
               if not data.get('code') or not data.get('userId'):
                   print(f'{consumer_name} received a msg {msg_id} without code or userId')
                   self.redis.xack(stream_name, consumer_group, msg_id)
                   continue

               code = data.get('code')
               user_id = data.get('userId')
               print(f'{consumer_name} received a msg {msg_id} with code {code} from user {user_id}')
               if target and target(code, data, msg_id, user_id):
                   # send ack when task has been scheduled
                   self.redis.xack(stream_name, consumer_group, msg_id)
               else:
                   # failed to schedule this task
                   task_report = {'msgId': msg_id, 'userId': user_id, 'status': 1}
                   self.redis.publish(REDIS_CHANNEL_FEEDBACK, json.dumps(task_report))
           except Exception as e:
               print("consumer is error: ", repr(e))
               # exception when schedule this task
               task_report = {'msgId': msg_id, 'userId': user_id, 'status': 2}
               self.redis.publish(REDIS_CHANNEL_FEEDBACK, json.dumps(task_report))

            # send response for debug without celery - Gavin !!!
           task_report = {'reqMsgId': msg_id, 'userId': user_id, 'retval': 123, 'status': 0}
           self.redis.publish(REDIS_CHANNEL_FEEDBACK, json.dumps(task_report))

    def msg_handler(self, code, data, msg_id, user_id):
        # print(f'msg data: {data}')
        try:
            # bind celery task id to msg_id
            task_id = msg_id + '@' + user_id
            match code:
                case '0':
                    func = None
                    alias = func.name+'@'+user_id
                    # task = add.apply_async(args=[1, 300000], task_id=task_id, shadow=alias)
                case '1':
                    func = None
                    alias = func.name + '@' + user_id
                    # task = sub.apply_async(args=[1, 9800000], task_id=task_id, shadow=alias)

                case _:
                    print(f'Unknown code: {code}')
                    return True

            print(f'Running task: {task_id}')
            time.sleep(10)
            # block mode if you get result here
            # print(f'Result: {task.get()}')
            return True
        except Exception as e:
            print(repr(e))
            return False