#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/2/15
# @File           : ray_tuner.py
# @desc           : train ml

import os
import tempfile
from typing import Dict
import matplotlib
import mlflow
import ray
import torch
import torchvision.datasets
from ray.air import ScalingConfig
from ray.train.torch import TorchConfig, TorchTrainer
from msgq.redis_client import RedisClient
from ray import tune, train
import ray.tune.search as search
from utils.ray.ray_reporter import RAY_EXPERIMENT_REPORT, RAY_JOB_EXCEPTION

TEMP_DIR = tempfile.mkdtemp()

@ray.remote
class PyTorchTrainer:
    def __init__(self, type: str):
        self.type = type.upper()
        self.engine = None
        self.dataset_df = None
        self.transformed_df = None
        self.train_data = None
        self.dataset = None
        self.trainset = None
        self.evalset = None
        self.testset = None

    # extract data from db or file system
    def extract(self, dataset_name: str):
        match self.type:
            case 'MYSQL':
                self.testset = None
            case 'PYTORCH':
                transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize((0.5,), (0.5,))])
                self.dataset = eval(f'torchvision.datasets.{dataset_name}')(TEMP_DIR, train=True, download=True, transform=transform)
        return self.dataset

    # transform/split/shuffle
    def transform(self, dataset, targets: [str] = list, ratio: int = 0.3, batch_size: int = 64, shuffle: bool = False):
        # Split the data into train and validation sets.
        train_set, val_set = torch.utils.data.random_split(dataset, [1-ratio, ratio])
        self.trainset = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
        self.evalset = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle)
        return {'train': self.trainset, 'eval': self.evalset}

    # train ML algo based on ray and mlflow
    def train(self, params: dict, train_cls, data: Dict):
        # use AWS S3/minio as artifact repository
        os.environ["AWS_ACCESS_KEY_ID"] = params.get('s3_id')
        os.environ["AWS_SECRET_ACCESS_KEY"] = params.get('s3_key')
        os.environ["MLFLOW_S3_IGNORE_TLS"] = 'true'
        # os.environ["AWS_DEFAULT_REGION"] = None
        mlflow.environment_variables.MLFLOW_S3_IGNORE_TLS = 'true'
        mlflow.environment_variables.MLFLOW_S3_ENDPOINT_URL = params.get('s3_url')
        # use mysql db as tracking store
        mlflow.set_tracking_uri(params['tracking_url'])
        # resolve warning 'Experiment state snapshotting has been triggered multiple times in the last 5.0 seconds'
        os.environ['TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S'] = '0'
        # to disable log deduplication
        os.environ["RAY_DEDUP_LOGS"] = "0"
        # enable custom ProgressReporter when set to 0
        os.environ['RAY_AIR_NEW_OUTPUT'] = '0'
        # os.environ["WORLD_SIZE"] = '1'
        # os.environ["RANK"] = '0'

        params['tune_param']['dist'] = True
        # check if current experiment exists
        exper = mlflow.get_experiment_by_name(params['exper_name'])
        if exper:
            params['exper_id'] = exper.experiment_id
        else:
            # create a new experiment
            exper_tags = {'org_id': params['org_id'], 'algo_id': params['algo_id'], 'algo_name': params['algo_name']}
            params['exper_id'] = mlflow.create_experiment(name=params['exper_name'], tags=exper_tags,
                                     artifact_location=params['artifact_location'])

        params['tune_param']['exper_id'] = params['exper_id']
        params['tune_param']['data'] = data
        mlflow.set_experiment(experiment_id=params['exper_id'])
        # resolve the warning 'Matplotlib GUI outside of the main thread will likely fail'
        matplotlib.use('agg')
        # mlflow.autolog()

        # Configure computation resources
        scaling_cfg = ScalingConfig(num_workers=1, use_gpu=True)
        torch_cfg = TorchConfig(backend="gloo")
        trainer = TorchTrainer(
            train_loop_per_worker=train_cls.train,
            scaling_config=scaling_cfg,
            torch_config=torch_cfg
        )

        # storage_path is not used because we are using mlflow to save result on S3
        # earlystop will cause run.status is still running and end_time will be null
        tune_cfg = tune.TuneConfig(num_samples=params['trials'],
                                   search_alg=search.BasicVariantGenerator(max_concurrent=3))
        run_cfg = train.RunConfig(name=params['exper_name'],  # stop=params.get('stop'),
                                  checkpoint_config=False, log_to_file=False, storage_path=None)

        report = {'userId': params['user_id'],
                  'payload': {'code': RAY_EXPERIMENT_REPORT, 'msg': '',
                              'data': {'algoId': params['algo_id'], 'experId': params['exper_id'], 'status': 1}}}
        RedisClient().feedback(report)
        tuner = tune.Tuner(trainable=trainer,
                           tune_config=tune_cfg,
                           run_config=run_cfg,
                           param_space={"train_loop_config": params['tune_param']})
        try:
            # start train......
            result = tuner.fit()
        except ValueError:
            exception = {'userId': params['user_id'],
                      'payload': {'code': RAY_JOB_EXCEPTION, 'msg': 'tuner.fit exception',
                                  'data': {'algoId': params['algo_id'], 'experId': params['exper_id']}}}
            # report['payload']['data']['detail'] = e
            print(exception)
            RedisClient().feedback(exception)
        else:
            report['payload']['data']['status'] = 0
            RedisClient().feedback(report)


