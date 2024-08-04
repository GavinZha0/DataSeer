import math
from datetime import datetime
from msgq.redis_client import RedisClient
from ray.tune import Callback

RAY_STEP_REPORT = 1
RAY_EPOCH_REPORT = 2
RAY_TRIAL_REPORT = 3
RAY_EXPERIMENT_REPORT = 4
RAY_JOB_REPORT = 5


class RayReport(Callback):
    def __init__(self, user_id, algo_id, exper_id, num_trials, num_epochs, metrics):
        self.user = user_id
        self.algo = algo_id
        self.exper = exper_id
        self.metrics = metrics
        self.now = datetime.now()
        self.completed_trials = 0
        self.total_trials = num_trials
        self.completed_epochs = 0
        self.total_epochs = num_trials * num_epochs

    # per step (not used)
    def on_step_end_unused(self, iteration, trials, **info):
        # report train progress (per 10 steps or per 30s)
        interval = datetime.now() - self.now
        if iteration % 10 == 0 or interval.total_seconds() > 30:
            self.now = datetime.now()
            report = {'userId': self.user, 'payload': {'code': RAY_STEP_REPORT, 'msg': '', 'data': {}}}
            payload_data = {'algoId': self.algo, 'experId': self.exper, 'step': iteration//10}
            report['payload']['data'] = payload_data
            print(report)
            RedisClient().feedback(report)

    # when get result of an epoch from ray.train.report()
    # training_iteration: The number of times train.report() has been called
    # so training_iteration = epoch
    # per epoch
    def on_trial_result(self, iteration, trials, trial, result, **info):
        print(f'...{iteration}')
        self.completed_epochs += 1
        progress = round(self.completed_epochs / self.total_epochs, 2)
        report = {'userId': self.user, 'payload': {'code': RAY_EPOCH_REPORT, 'msg': '', 'data': {}}}
        payload_data = {'algoId': self.algo, 'experId': self.exper, 'trialId': trial.trial_id,
                        'epoch': result.get('training_iteration'), 'progress': progress,
                        'params': trial.evaluated_params}
        report['payload']['data'] = payload_data

        if result.get('time_total_s'):
            payload_data['duration'] = math.ceil(result.get('time_total_s'))

        evaluation: dict = {}
        if self.metrics:
            for kpi_name in self.metrics:
                evaluation[kpi_name] = result.get(kpi_name)
            payload_data['eval'] = evaluation

        print(report)
        RedisClient().feedback(report)

    # per trial
    def on_trial_complete(self, iteration, trials, trial, **info):
        self.completed_trials += 1
        progress = round(self.completed_trials/self.total_trials, 2)
        report = {'userId': self.user, 'payload': {'code': RAY_TRIAL_REPORT, 'msg': '', 'data': {}}}
        payload_data = {'algoId': self.algo, 'experId': self.exper, 'trialId': trial.trial_id,
                        'params': trial.evaluated_params,
                        'duration': math.ceil(trial.last_result.get('time_total_s')),
                        'progress': progress}
        report['payload']['data'] = payload_data

        evaluation: dict = {}
        if self.metrics:
            for kpi_name in self.metrics:
                evaluation[kpi_name] = trial.last_result.get(kpi_name)
            payload_data['eval'] = evaluation

        print(report)
        RedisClient().feedback(report)

    # per experiment
    def on_experiment_end(self, trials, **info):
        report = {'userId': self.user, 'payload': {'code': RAY_EXPERIMENT_REPORT, 'msg': '', 'data': {}}}
        payload_data = {'algoId': self.algo, 'experId': self.exper, 'trial': []}
        report['payload']['data'] = payload_data

        for trial in trials:
            trial_info = {'id': trial.trial_id, 'params': trial.evaluated_params, 'eval': {}}
            if trial.get_error():
                trial_info['error'] = trial.get_error()

            evaluation: dict = {}
            if self.metrics:
                for kpi_name in self.metrics:
                    evaluation[kpi_name] = trial.last_result.get(kpi_name)
                trial_info['eval'] = evaluation
            payload_data['trial'].append(trial_info)
        print(report)
        RedisClient().feedback(report)

