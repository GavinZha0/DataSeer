import string

sklearn_setup_mlflow = '''
    mlflow.set_tracking_uri(config.get('tracking_url'))
    mlflow.set_experiment(experiment_id=config.get('exper_id'))
    mlflow.sklearn.autolog(extra_tags={MLFLOW_RUN_NAME: ray.train.get_context().get_trial_id(), MLFLOW_USER: config.get('user_id')})
    matplotlib.use('agg')
'''

sklearn_tpl = string.Template('''
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

pytorch_tpl = string.Template('''
import os
import mlflow
from ray.air.integrations.mlflow import setup_mlflow
from ray.train.lightning import prepare_trainer, RayTrainReportCallback
from ray.train.torch import prepare_data_loader, prepare_model
from config.settings import BASE_DIR


${CUSTOM_NET_CLASS}


# for train of custom pytorch module
class CustomTrain:
  def train(config: dict):
    model = CustomNet(config)
    # use 'pop' to avoid saving data into mlflow parameters
    data = config.pop('data')
    train_loader = data['train']
    eval_loader = data['eval']

    if config.get('dist'):
      train_loader = prepare_data_loader(train_loader)
      eval_loader = prepare_data_loader(eval_loader)
      model = prepare_model(model)

    torch.set_float32_matmul_precision('medium')
    # config will be saved into mlflow
    myflow = setup_mlflow(config, experiment_id=config.pop('exper_id'), tracking_uri=config.pop('tracking_url'))
    # extra_tag doesn't work
    myflow.autolog()
    myflow.pytorch.autolog()

    trainer = pl.Trainer(
      max_epochs=config.get("epochs", 1),
      devices="auto",
      accelerator="auto",
      enable_checkpointing=False,
      default_root_dir=os.path.join(BASE_DIR, "logs"),
      callbacks=[RayTrainReportCallback()],
      enable_progress_bar=False
    )
    # trainer = prepare_trainer(trainer)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=eval_loader)
''')


async def build_ray_trainable(framework: str, code: str, params: dict):
    match(framework):
        case 'sklearn':
            trainable_code = code.replace('setup_mlflow()', sklearn_setup_mlflow)
        case 'pytorch':
            trainable_code = pytorch_tpl.substitute({'CUSTOM_NET_CLASS': code})

    try:
        # save trainable class to local file
        with open(f'./temp/{params["module_name"]}.py', 'w') as file:
            file.write(trainable_code)
    except:
        return False

    return True
