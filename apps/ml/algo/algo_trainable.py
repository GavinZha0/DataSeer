import os.path
import string
from config.settings import TEMP_DIR

sklearn_library = '''
import ray
import mlflow
import matplotlib
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_USER
'''

sklearn_setup = '''
    mlflow.set_tracking_uri(config.get('tracking_url'))
    mlflow.set_experiment(experiment_id=config.get('exper_id'))
    mlflow.sklearn.autolog(extra_tags={MLFLOW_RUN_NAME: ray.train.get_context().get_trial_name(), MLFLOW_USER: config.get('user_id')})
    matplotlib.use('agg')
    train_y = val_x = val_y = None
'''

sklearn_tpl = string.Template('''
import ray
import mlflow
import matplotlib
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_USER
from sklearn.{MODULE} import {ALGORITHM}
from sklearn import metrics

# for sklearn algo
class CustomTrain:
    def train(config: dict, data: dict):
        mlflow.set_tracking_uri(config.get('tracking_url'))
        mlflow.set_experiment(experiment_id=config.get('exper_id'))
        mlflow.sklearn.autolog(extra_tags={MLFLOW_RUN_NAME: ray.train.get_context().get_trial_name(), MLFLOW_USER: config.get('user_id')})
        matplotlib.use('agg')

        estimator = {ALGORITHM}({PARAMS})
        for epoch in range(config.get("epochs", 1)):
            estimator.fit(data['x'], data['y'])
            {SCORE_NAME}_fn = metrics.get_scorer('{SCORE_NAME}')
            {SCORE_NAME} = {SCORE_NAME}_fn(estimator, data['x'], data['y'])
            ray.train.report({"{SCORE_NAME}": {SCORE_NAME}})
''')

pytorch_tpl_bak = string.Template('''
import os
import ray
import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_USER, MLFLOW_RUN_NAME
from ray.train.lightning import prepare_trainer, RayTrainReportCallback
from ray.train.torch import prepare_data_loader, prepare_model
from config.settings import BASE_DIR
import torch

${PL_MODULE_CODE}

# trainable function of Ray
class RayTrainable:
  def train(config: dict):
    # build model with config parameters
    model = ${PL_MODULE_CLASS}(config)

    # use 'pop' to avoid saving data into mlflow parameters
    # data is TensorDataset
    tensor_data = config.pop('data')
    # split data into train and validation before training
    train_set, val_set = torch.utils.data.random_split(tensor_data, [1 - config.get('ratio'), config.get('ratio')], generator=torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=config.get('shuffle'))
    eval_loader = torch.utils.data.DataLoader(val_set, batch_size=config.get('batch'), shuffle=False)


    # distribution
    if config.get('dist'):
      train_loader = prepare_data_loader(train_loader)
      eval_loader = prepare_data_loader(eval_loader)
      model = prepare_model(model)

    # You are using a CUDA device that has Tensor Cores. You should set `set_float32_matmul_precision('medium' | 'high')`
    torch.set_float32_matmul_precision('medium')

    # mlflow setup
    mlflow.set_tracking_uri(config.get('tracking_url'))
    mlflow.set_experiment(experiment_id=config.get('exper_id'))
    mlflow.pytorch.autolog(extra_tags={MLFLOW_RUN_NAME: ray.train.get_context().get_trial_name(), MLFLOW_USER: config.get('user_id')})

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


pytorch_tpl = string.Template('''
import os
import ray
import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_USER, MLFLOW_RUN_NAME
from ray.train.lightning import prepare_trainer, RayTrainReportCallback
from ray.train.torch import prepare_data_loader, prepare_model
from config.settings import BASE_DIR
import torch
from typing import Any, Dict
import numpy as np

${PL_MODULE_CODE}

# trainable function of Ray
class RayTrainable:
  def train(config: dict):
    # build model with config parameters
    model = ${PL_MODULE_CLASS}(config)

    # get data by way 1
    data_shard = ray.train.get_dataset_shard("train_set")
    train_set = None
    def collate_fn(batch: Dict[str, np.ndarray]) -> Any:
        # return tensor array
        # [(tensor([3.5000, 0.2000, 5.1000, 1.4000]), tensor(0))...]
        return batch['item']
    for batch in data_shard.iter_torch_batches(collate_fn=collate_fn):
        if train_set is None:
            train_set = batch
        else:
            train_set += batch
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.get('batch'), shuffle=False)

    # get data by way 2
    data_shard = ray.train.get_dataset_shard("val_set")
    val_set = []
    for batch in data_shard.iter_batches(batch_size=1):
        val_set.append(batch['item'][0])
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.get('batch'), shuffle=False)

    # You are using a CUDA device that has Tensor Cores. You should set `set_float32_matmul_precision('medium' | 'high')`
    torch.set_float32_matmul_precision('medium')

    # mlflow setup
    mlflow.set_tracking_uri(config.get('tracking_url'))
    mlflow.set_experiment(experiment_id=config.get('exper_id'))
    mlflow.pytorch.autolog(extra_tags={MLFLOW_RUN_NAME: ray.train.get_context().get_trial_name(), MLFLOW_USER: config.get('user_id')})

    trainer = pl.Trainer(
      max_epochs=config.get("epochs", 1),
      devices="auto",
      accelerator="auto",
      enable_checkpointing=False,
      default_root_dir=os.path.join(BASE_DIR, "logs"),
      callbacks=[RayTrainReportCallback()],
      enable_progress_bar=False,
      strategy='auto'
    )

    # distribution
    if config.get('dist'):
      train_loader = prepare_data_loader(train_loader)
      val_loader = prepare_data_loader(val_loader)
      model = prepare_model(model)
      # trainer = prepare_trainer(trainer)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
''')


async def build_ray_trainable(category: str, code: str, params: dict):
    # category = 'sklearn.classifier'
    temp_cat = category.split('.')
    frame = temp_cat[0].upper()

    if frame == 'SKLEARN':
        trainable_code = code.replace('import ray', sklearn_library)
        trainable_code = trainable_code.replace('train_y = val_x = val_y = None', sklearn_setup)
    elif frame == 'PYTORCH':
        trainable_code = pytorch_tpl.substitute({'PL_MODULE_CODE': code, 'PL_MODULE_CLASS': params['cls_name'][0]})

    try:
        # save trainable class to local file
        if os.path.exists(f'{TEMP_DIR}/ml/') is False:
            os.mkdir(f'{TEMP_DIR}/ml/')
        # with open(f'{TEMP_DIR}/ml/{params["module_name"]}.py', 'w') as file:
        #    file.write(trainable_code)
    except Exception as e:
        print(e)
        return False

    return True



# for pytorch lightning trainable function
# myflow's extra_tag doesn't work, but mlflow works. Is mlflow global instance? Will it impact others?
# and you will see error about 'unsupported autologging version', but works
# myflow = setup_mlflow(experiment_id=config.pop('exper_id'), tracking_uri=config.pop('tracking_url'))
# myflow.pytorch.autolog(extra_tags={MLFLOW_USER: 3})
# mlflow.set_tracking_uri(config.get('tracking_url'))
# mlflow.set_experiment(experiment_id=config.get('exper_id'))
# mlflow.pytorch.autolog(extra_tags={MLFLOW_RUN_NAME: ray.train.get_context().get_trial_name(), MLFLOW_USER: 3})

# for sklearn
# setup_mlflow doesn't work, so have to use global mlflow