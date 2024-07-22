import string

sk_tpl = string.Template('''
import ray
from ray.air.integrations.mlflow import setup_mlflow
from sklearn.ensemble import ${SKCLASS}
from sklearn.metrics import accuracy_score

# for sklearn algo
class CustomAlgo:
    def train(config: dict):
        setup_mlflow(
            config,
            experiment_id=config.get("experiment_id", None),
            experiment_name=config.get("experiment_name", None),
            tracking_uri=config.get("tracking_uri", None),
            artifact_location=config.get("artifact_location", None),
            create_experiment_if_not_exists=True,
            run_name=config.get("run_name", None),
            tags=config.get("tags", None)
        )

        model = ${ALGO}(${ARGS})
        for epoch in range(config.get("epochs", 1)):
            model.fit(config['x'], config['y'])
            y_predict = model.predict(config['x'])
            acc = accuracy_score(config['y'], y_predict)
            ray.train.report({'acc': acc})
''')

def build_sk_train_class(params: dict):
    tpl = sk_tpl
