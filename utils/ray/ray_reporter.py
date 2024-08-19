import math
from datetime import datetime
from msgq.redis_client import RedisClient
from ray.tune import Callback

RAY_STEP_REPORT = 1
RAY_EPOCH_REPORT = 2
RAY_TRIAL_REPORT = 3
RAY_EXPERIMENT_REPORT = 4
RAY_JOB_EXCEPTION = 5

JOB_PROGRESS_START = 1
JOB_PROGRESS_END = 100


class RayReport(Callback):
    def __init__(self, params: list):
        self.user = params.get('user_id')
        self.algo = params.get('algo_id')
        self.name = params.get('algo_name')
        self.exper = params.get('exper_id')
        self.score = params.get('score')
        self.now = datetime.now()
        self.completed_trials = 0
        self.total_trials = params.get('trials')
        self.completed_epochs = 0
        self.total_epochs = params.get('trials') * params.get('epochs')

    # start a new experiment
    # use experimentProgress to report START
    def setup_X(self, stop, num_samples, total_num_samples, **info):
        # progress 1: experiment start
        report = dict(uid=self.user, code=RAY_EXPERIMENT_REPORT, msg='',
                      data=dict(name=self.name, algoId=self.algo, experId=self.exper, progress=1))
        RedisClient().feedback(report)
        RedisClient().notify(report)

    # per step (not used)
    def on_step_end_X(self, iteration, trials, **info):
        # report train progress (per 10 steps or per 30s)
        interval = datetime.now() - self.now
        if iteration % 10 == 0 or interval.total_seconds() > 30:
            self.now = datetime.now()
            report = dict(uid=self.user, code=RAY_STEP_REPORT, msg='',
                          data=dict(name=self.name, algoId=self.algo, experId=self.exper, step=iteration // 10))
            print(report)
            RedisClient().feedback(report)
            RedisClient().notify(report)

    # when get result of an epoch from ray.train.report()
    # training_iteration: The number of times train.report() has been called
    # so training_iteration = epoch
    # per epoch
    def on_trial_result(self, iteration, trials, trial, result, **info):
        self.completed_epochs += 1
        progress = round(self.completed_epochs / self.total_epochs, 2) * 100
        report = dict(uid=self.user, code=RAY_EPOCH_REPORT, msg='',
                      data=dict(name=self.name, algoId=self.algo, experId=self.exper, trialId=trial.trial_id, progress=progress,
                                epoch=result.get('training_iteration'), params=trial.evaluated_params))
        report_data = report['data']

        if result.get('time_total_s'):
            report_data['duration'] = math.ceil(result.get('time_total_s'))

        if self.score:
            # user specified eval metrics
            report_data['score'] = result.get(self.score)

        print(report)
        RedisClient().feedback(report)
        RedisClient().notify(report)

    # per trial
    def on_trial_complete(self, iteration, trials, trial, **info):
        self.completed_trials += 1
        progress = round(self.completed_trials / self.total_trials, 2) * 100
        report = dict(uid=self.user, code=RAY_TRIAL_REPORT, msg='',
                      data=dict(name=self.name, algoId=self.algo, experId=self.exper, trialId=trial.trial_id,
                                params=trial.evaluated_params, progress=progress,
                                duration=math.ceil(trial.last_result.get('time_total_s'))))
        report_data = report['data']

        if self.score:
            # user specified eval metrics
            report_data['score'] = trial.last_result.get(self.score)

        print(report)
        # RedisClient().feedback(report)
        # RedisClient().notify(report)

    # per experiment
    # use experimentProgress to report end
    def on_experiment_end(self, trials, **info):
        # progress 100: experiment end
        report = dict(uid=self.user, code=RAY_EXPERIMENT_REPORT, msg='',
                      data=dict(name=self.name, algoId=self.algo, experId=self.exper, progress=100, trials=[]))
        report_data = report['data']

        for trial in trials:
            trial_info = dict(id=trial.trial_id, params=trial.evaluated_params)
            if trial.get_error():
                trial_info['error'] = trial.get_error()

            if self.score:
                # user specified eval metrics
                trial_info['score'] = trial.last_result.get(self.score)
            report_data['trials'].append(trial_info)
        print(report)
        # RedisClient().feedback(report)
        # RedisClient().notify(report)

    def experimentProgress(self, progress: int):
        # progress 1: experiment start
        # progress 100: experiment end
        report = dict(uid=self.user, code=RAY_EXPERIMENT_REPORT, msg='',
                      data=dict(name=self.name, algoId=self.algo, experId=self.exper, progress=progress))
        RedisClient().feedback(report)
        RedisClient().notify(report)

    def experimentException(self, exception: str):
        report = dict(uid=self.user, code=RAY_JOB_EXCEPTION, msg='tuner.fit exception',
                      data=dict(name=self.name, algoId=self.algo, experId=self.exper, detail=exception))
        RedisClient().feedback(report)
        RedisClient().notify(report)
