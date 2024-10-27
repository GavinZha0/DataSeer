#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/2/15
# @File           : ray_tuner.py
# @desc           : train ml

import os
from typing import Dict
import mlflow
import pandas as pd
import ray
import ray.tune.search as search
import ray.tune.schedulers as schedule
from pandas import CategoricalDtype
from ray import tune, train
from ray.exceptions import RayError
from ray.tune.experimental.output import get_air_verbosity, AirVerbosity
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sklearn import preprocessing as pp
from sklearn import feature_extraction as fe
from config.settings import TEMP_DIR
from utils.ray.ray_reporter import RayReport, JOB_PROGRESS_START, JOB_PROGRESS_END
from ray.air import ScalingConfig
from ray.train.torch import TorchConfig, TorchTrainer


@ray.remote
class RayTrainer:
    def __init__(self, frame: str):
        self.frame = frame.upper()

    # train ML algo based on ray and mlflow
    def train(self, params: dict, train_func, data: Dict):
        match self.frame:
            case 'SKLEARN':
                return self.trainSk(params, train_func, data)
            case 'PYTORCH':
                return self.trainTorch(params, train_func, data)


    # train ML algo based on ray and mlflow
    def trainSk(self, params: dict, train_func, data: Dict):
        # use AWS S3/minio as artifact repository
        os.environ["AWS_ACCESS_KEY_ID"] = params.get('s3_id')
        os.environ["AWS_SECRET_ACCESS_KEY"] = params.get('s3_key')
        mlflow.environment_variables.MLFLOW_S3_ENDPOINT_URL = params.get('s3_url')
        # use mysql db as tracking store
        mlflow.set_tracking_uri(params['tracking_url'])
        # resolve warning 'Experiment state snapshotting has been triggered multiple times in the last 5.0 seconds'
        os.environ['TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S'] = '0'
        # to disable log deduplication
        os.environ["RAY_DEDUP_LOGS"] = "0"
        # enable custom ProgressReporter when set to 0
        os.environ['RAY_AIR_NEW_OUTPUT'] = '0'

        # create a new experiment with UNIQUE name for mlflow (ex: algo_3_1234567890)
        exper_tags = dict(org_id=params['org_id'], algo_id=params['algo_id'], algo_name=params['algo_name'],
                          user_id=params['user_id'])
        if params.get('args'):
            exper_tags['args'] = '|'.join(params['args'])

        params['exper_id'] = mlflow.create_experiment(name=params['exper_name'], tags=exper_tags,
                                                      artifact_location=params['artifact_location'])
        params['tune_param']['exper_id'] = params['exper_id']
        # resolve the warning 'Matplotlib GUI outside of the main thread will likely fail'
        # matplotlib.use('agg')
        # mlflow.autolog()

        # create progress report
        progressRpt = RayReport(params)
        progressRpt.experimentProgress(JOB_PROGRESS_START)

        if params.get('gpu') == True:
            tune_func = tune.with_resources(tune.with_parameters(train_func, data=data), resources={"gpu": 1})
        else:
            tune_func = tune.with_parameters(train_func, data=data)

        tune_cfg = tune.TuneConfig(num_samples=params['trials'],
                                   search_alg=search.BasicVariantGenerator(max_concurrent=1),
                                   scheduler=schedule.ASHAScheduler(mode="max"),
                                   time_budget_s=params['timeout'] * 60 * params['trials'] if params.get('timeout') else None)
        # ray will save tune results into storage_path with sub-folder exper_name
        # this is not used because we are using mlflow to save result on S3
        run_cfg = train.RunConfig(name=params['exper_name'],  stop=params.get('stop'),
                                  verbose=get_air_verbosity(AirVerbosity.DEFAULT),
                                  log_to_file=False, storage_path=TEMP_DIR+'/tune/',
                                  checkpoint_config=False,
                                  callbacks=[progressRpt])

        tuner = tune.Tuner(trainable=tune_func,
                           tune_config=tune_cfg,
                           run_config=run_cfg,
                           param_space=params['tune_param'])

        try:
            # start train......
            result = tuner.fit()
        except RayError as e:
            print(e)
            # report exception
            progressRpt.experimentException(e)
        else:
            # report progress
            progressRpt.experimentProgress(JOB_PROGRESS_END)



    # train ML algo based on ray and mlflow
    def trainTorch(self, params: dict, train_func, data: Dict):
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

        # create a new experiment with UNIQUE name for mlflow (ex: algo_3_1234567890)
        exper_tags = dict(org_id=params['org_id'], algo_id=params['algo_id'], algo_name=params['algo_name'],
                              user_id=params['user_id'], args='|'.join(params['args']))
        params['exper_id'] = mlflow.create_experiment(name=params['exper_name'], tags=exper_tags,
                                                          artifact_location=params['artifact_location'])

        params['tune_param']['exper_id'] = params['exper_id']
        params['tune_param']['data'] = data
        # resolve the warning 'Matplotlib GUI outside of the main thread will likely fail'
        # matplotlib.use('agg')
        # mlflow.autolog()

        # create progress report
        progressRpt = RayReport(params)
        progressRpt.experimentProgress(JOB_PROGRESS_START)

        # Configure computation resources
        scaling_cfg = ScalingConfig(num_workers=1, use_gpu=True)
        torch_cfg = TorchConfig(backend="gloo")
        trainer = TorchTrainer(
            train_loop_per_worker=train_func,
            scaling_config=scaling_cfg,
            torch_config=torch_cfg
        )

        # ray will save tune results into storage_path with sub-folder exper_name
        # this is not used because we are using mlflow to save result on S3
        # earlystop will cause run.status is still running and end_time will be null
        tune_cfg = tune.TuneConfig(num_samples=params['trials'],
                                   search_alg=search.BasicVariantGenerator(max_concurrent=3))
        run_cfg = train.RunConfig(name=params['exper_name'],  # stop=params.get('stop'),
                                    checkpoint_config=False, log_to_file=False, storage_path=TEMP_DIR + '/tune/',
                                    callbacks=[progressRpt])

        tuner = tune.Tuner(trainable=trainer,
                            tune_config=tune_cfg,
                           run_config=run_cfg,
                           param_space={"train_loop_config": params['tune_param']})
        try:
            # start train......
            result = tuner.fit()
        except RayError as e:
            print(e)
            progressRpt.experimentException(e)
        else:
            # report progress
            progressRpt.experimentProgress(JOB_PROGRESS_END)
