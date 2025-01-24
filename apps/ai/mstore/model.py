import os
import random
import socket
import subprocess
import time

import mlflow
import psutil
import ray
import requests
from config import settings

AI_MODEL_STATUS_IDLE = 0
AI_MODEL_STATUS_SERVING = 1
AI_MODEL_STATUS_EXCEPTION = 2
AI_MODEL_STATUS_UNKNOWN = 3

DEFAULT_API_PORT = 7788
START_API_PORT = 6000
END_API_PORT = 9000

async def get_model_schema(run_id: str):
    os.environ["AWS_ACCESS_KEY_ID"] = settings.AWS_S3_ACCESS_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.AWS_S3_SECRET_KEY
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.AWS_S3_ENDPOINT

    mlflow.set_tracking_uri(settings.SQLALCHEMY_MLFLOW_DB_URL)
    model_info = mlflow.models.get_model_info(f"runs:/{run_id}/model")
    model_sign:mlflow.models.signature.ModelSignature = model_info.signature

    return dict(inputs=model_sign.inputs.to_dict(), outputs=model_sign.outputs.to_dict())



async def model_deploy(run_id: str, platform: str, endpoint: str, img_name: str):
    os.environ["AWS_ACCESS_KEY_ID"] = settings.AWS_S3_ACCESS_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.AWS_S3_SECRET_KEY
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.AWS_S3_ENDPOINT
    os.environ["MLFLOW_TRACKING_URI"] = settings.SQLALCHEMY_MLFLOW_DB_URL

    # http://127.0.0.1:7788/invocations
    # get port from endpoint
    ipport = None
    svr_port = None
    if endpoint:
        # has endpoint
        segs = endpoint.split('/')
        ipport = segs[2].split(':')

    if ipport and len(ipport) > 1:
        # check specific ip and port
        svr_ip = ipport[0]
        svr_port = int(ipport[1])
    elif ipport and len(ipport) > 0:
        svr_ip = ipport[0]


    if svr_port:
        if port_idle(svr_port) is False:
            if svr_port == DEFAULT_API_PORT:
                # allocate a port if default port is occupied
                svr_port = allocate_port()
            else:
                # user defined port is occupied
                return False, 'Target port is occupied!'
    else:
        # get a random port when no endpoint or specific port
        svr_port = allocate_port()

    url = 'Failed to start service'
    if platform and platform == 'Ray':
        cmd = f'mlflow models serve -m runs:/{run_id}/model -p {svr_port} --env-manager=local'
        rst = subprocess.Popen(cmd, shell=True)
    elif platform and platform == 'K8s':
        cmd = f'mlflow models serve -m runs:/{run_id}/model -p {svr_port} --env-manager=local'
        rst = subprocess.Popen(cmd, shell=True)
    elif platform and platform == 'Docker':
        build_docker_img.remote(run_id, img_name)
        url = f'({img_name})->http://ip:8080/invocations'
    else:
        svr_ip = socket.gethostbyname(socket.gethostname())
        cmd = f'mlflow models serve -m runs:/{run_id}/model -h {svr_ip} -p {svr_port} --env-manager=local'
        rst = subprocess.Popen(cmd, shell=True)
        # endpoint url
        url = f'http://{svr_ip}:{svr_port}/invocations'

    # check if it is working
    return check_endpoint(platform, url), url

async def model_terminate(platform: str, endpoint: str):
    # http://127.0.0.1:7788/invocations
    # get port from rest api
    svr_port = None
    if endpoint:
        segs = endpoint.split('/')
        ipport = segs[2].split(':')
        if ipport and len(ipport) > 1:
            # check specific port
            svr_port = int(ipport[1])
            if port_idle(svr_port):
                # port is idle
                return
        else:
            # no port is in endpoint
            return
    else:
        # endpoint is null
        return

    # find pid by port
    pid = None
    connections = psutil.net_connections()
    for con in connections:
        if con.raddr != tuple():
            if con.raddr.port == svr_port:
                pid = con.pid
                break
        if con.laddr != tuple():
            if con.laddr.port == svr_port:
                pid = con.pid
                break

    if pid is None:
        # can't find pid
        return

    if platform and platform == 'ray':
        # terminate pid
        if pid:
            p = psutil.Process(pid)
            p.terminate()
    else:
        # terminate pid
        if pid:
            p = psutil.Process(pid)
            p.terminate()


def port_idle(port: int, ip: str = None):
    svr_port = int(port)
    svr_ip = ip
    if ip is None:
        svr_ip = socket.gethostbyname(socket.gethostname())

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = s.connect_ex((svr_ip, svr_port))
        s.close()
        if result == 0:
            # it is occupied
            return False
        else:
            # it is idle
            return True
    except:
        return True

def allocate_port():
    port = random.randrange(6000, 9000)
    if port_idle(port):
        return port
    else:
        return allocate_port()


def check_endpoint(platform: str, url: str):
    # default unknown
    serving = AI_MODEL_STATUS_UNKNOWN

    # MLflow, Ray, ...
    match platform.lower():
        case 'mlflow':
            rest_api = url.replace('invocations', 'version')
            for i in range(5):
                time.sleep(3)
                try:
                    result = requests.get(url=rest_api)
                    if result.content and len(result.content) > 0:
                        # it is serving
                        return AI_MODEL_STATUS_SERVING
                    else:
                        # it is not working
                        return AI_MODEL_STATUS_EXCEPTION
                except:
                    serving = AI_MODEL_STATUS_EXCEPTION
                    print(f'Ping-{i}: {rest_api}')
        case 'ray':
            return AI_MODEL_STATUS_SERVING
    return serving


@ray.remote
def build_docker_img(run_id: str, img_name: str):
    os.environ["AWS_ACCESS_KEY_ID"] = settings.AWS_S3_ACCESS_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.AWS_S3_SECRET_KEY
    mlflow.environment_variables.MLFLOW_S3_ENDPOINT_URL = settings.AWS_S3_ENDPOINT
    mlflow.set_tracking_uri(settings.SQLALCHEMY_MLFLOW_DB_URL)
    mlflow.models.build_docker(model_uri=f"runs:/{run_id}/model", name=f"{img_name}", enable_mlserver=True)
