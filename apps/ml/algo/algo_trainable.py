import os.path
import string

from mlflow import xgboost

from config.settings import TEMP_DIR

mlflow_library = '''
import ray
import mlflow
import matplotlib
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_USER
'''

mlflow_setup = '''
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

pytorch_data_tpl = string.Template('''
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

    # get dataset of ray
    data_shard = ray.train.get_dataset_shard("train")
    train_df = data_shard.materialize().to_pandas()
    data_shard = ray.train.get_dataset_shard("validation")
    val_df = data_shard.materialize().to_pandas()
    if config['targets']:
        if train_df.dtypes[config['targets'][0]].name.startswith('float'):
            train_set = torch.utils.data.TensorDataset(torch.Tensor(train_df.drop(columns=config['targets']).values),
                                                      torch.Tensor(train_df[config['targets']].to_numpy().ravel()))
            val_set = torch.utils.data.TensorDataset(torch.Tensor(val_df.drop(columns=config['targets']).values),
                                                       torch.Tensor(val_df[config['targets']].to_numpy().ravel()))
        else:
            train_set = torch.utils.data.TensorDataset(torch.Tensor(train_df.drop(columns=config['targets']).values),
                                                       torch.LongTensor(train_df[config['targets']].to_numpy().ravel()))
            val_set = torch.utils.data.TensorDataset(torch.Tensor(val_df.drop(columns=config['targets']).values),
                                                     torch.LongTensor(val_df[config['targets']].to_numpy().ravel()))
    else:
        train_set = torch.utils.data.TensorDataset(torch.Tensor(train_df.values))
        val_set = torch.utils.data.TensorDataset(torch.Tensor(val_df.values))

    # split data into train and validation before training
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.get('batch_size'), shuffle=False,
                                            num_workers=2, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.get('batch_size'), shuffle=False, num_workers=2,
                                            persistent_workers=True)

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

pytorch_lstm_tpl = string.Template('''
import os
import ray
import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_USER, MLFLOW_RUN_NAME
from ray.train.lightning import prepare_trainer, RayTrainReportCallback
from ray.train.torch import prepare_data_loader, prepare_model
from config.settings import BASE_DIR
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

${PL_MODULE_CODE}

class RayTrainable:
  def train(config: dict):
    # build model with config parameters
    model = ${PL_MODULE_CLASS}(config)

    input_size = config['input_size']
    output_size = config['output_size']
    seq_len = config['seq_len']
    gap_len = config['gap_len']

    # get dataset of ray and convert to pandas dataframe
    # DataIterator(MaterializedDataset(num_blocks=1, num_rows=144, schema={month: string, passengers: float64}))
    data_shard = ray.train.get_dataset_shard("train")
    # for time series data, normally there are only two columns (datetime and value)
    df = data_shard.materialize().to_pandas()
    # get target columns for LSTM training (datetime is unused)
    df = pd.DataFrame(df, columns=['passengers'], dtype=float)
    dv = df.to_numpy()

    # please consider test size when split data in order to avoid data waste
    # (seq_len * input_size) features are used to predict following output_size target values
    # e.g. 4x12 month data is used to predict next 12 months
    # gap_len is the gap between feature n and feature n+1
    end_pos = len(dv) - seq_len * input_size - output_size
    start_pos = end_pos % gap_len
    # extract features and labels from time series.
    train_x = [dv[i:i+seq_len*input_size].reshape(-1, input_size) for i in range(start_pos, end_pos+1, gap_len)]
    train_y = [dv[i:i+output_size].T.squeeze() for i in range(start_pos+seq_len*input_size, end_pos+1+seq_len*input_size, gap_len)]
    # rows = (end_pos//gap_len)+1
    # x shape: (rows, seq_len, input_size)
    train_x = np.array(train_x)
    # y shape: (rows, output_size)
    train_y = np.array(train_y).reshape(-1, output_size)

    # do same operations for validation data
    data_shard = ray.train.get_dataset_shard("validation")
    df = data_shard.materialize().to_pandas()
    df = pd.DataFrame(df, columns=['passengers'], dtype=float)
    dv = df.to_numpy()
    end_pos = len(dv) - seq_len * input_size - output_size
    start_pos = end_pos % gap_len
    val_x = [dv[i:i+seq_len*input_size].reshape(-1, input_size) for i in range(start_pos, end_pos+1, gap_len)]
    val_y = [dv[i:i+output_size].T.squeeze() for i in range(start_pos + seq_len * input_size, end_pos+1+seq_len*input_size, gap_len)]
    val_x = np.array(val_x)
    val_y = np.array(val_y).reshape(-1, output_size)

    # convert to tensor and build data loader
    train_loader = DataLoader(TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y)), batch_size=config.get('batch_size'))
    val_loader = DataLoader(TensorDataset(torch.Tensor(val_x), torch.Tensor(val_y)), batch_size=config.get('batch_size'))

    # You are using a CUDA device that has Tensor Cores. You should set `set_float32_matmul_precision('medium' | 'high')`
    torch.set_float32_matmul_precision('medium')

    # mlflow setup
    mlflow.set_tracking_uri(config.pop('tracking_url'))
    mlflow.set_experiment(experiment_id=config.get('exper_id'))
    # save config parameters but autolog will not take effect. run without end time. how to fix?
    # mlflow.log_params(config)
    mlflow.pytorch.autolog(
        extra_tags={MLFLOW_RUN_NAME: ray.train.get_context().get_trial_name(), MLFLOW_USER: config.get('user_id')})


    trainer = pl.Trainer(
        max_epochs=config.get("epochs"),
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



pytorch_img_tpl = string.Template('''
import os
import ray
import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_USER, MLFLOW_RUN_NAME
from ray.train.lightning import prepare_trainer, RayTrainReportCallback
from ray.train.torch import prepare_data_loader, prepare_model
from config.settings import BASE_DIR
import torch
import numpy as np

${PL_MODULE_CODE}

# trainable function of Ray
class RayTrainable:
  def train(config: dict):
    # build model with config parameters
    model = ${PL_MODULE_CLASS}(config)

    # get dataset of ray
    data_shard = ray.train.get_dataset_shard("train")
    # MaterializedDataset(num_blocks=2,num_rows=60000,schema={item: extension<ray.data.arrow_pickled_object<ArrowPythonObjectType>>})
    train_set = []
    for item in data_shard.iter_batches(batch_size=1):
      train_set.append(item['item'][0])

    data_shard = ray.train.get_dataset_shard("validation")
    val_set = []
    for item in data_shard.iter_batches(batch_size=1):
      val_set.append(item['item'][0])

    # split data into train and validation before training
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.get('batch_size'), shuffle=False,
                                            num_workers=2, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.get('batch_size'), shuffle=False, num_workers=2,
                                            persistent_workers=True)

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

async def build_ray_trainable(category: str, algo: str, code: str, params: dict):
    # e.g., category = 'sklearn.classifier' or 'boost.xgboost'
    # e.g., algo = 'linear_model.LogisticRegression' or 'XGBClassifier'
    cat = category.upper()
    trainable_code = ''

    if cat.startswith('SKLEARN') or cat.startswith('BOOST'):
        trainable_code = code.replace('import ray', mlflow_library)
        trainable_code = trainable_code.replace('train_y = val_x = val_y = None', mlflow_setup)
    elif cat.startswith('PYTORCH'):
        if cat.endswith('VISION'):
            trainable_code = pytorch_img_tpl.substitute({'PL_MODULE_CODE': code, 'PL_MODULE_CLASS': params['cls_name'][0]})
        elif algo.upper().endswith('LSTM'):
            trainable_code = pytorch_lstm_tpl.substitute({'PL_MODULE_CODE': code, 'PL_MODULE_CLASS': params['cls_name'][0]})
        else:
            trainable_code = pytorch_data_tpl.substitute({'PL_MODULE_CODE': code, 'PL_MODULE_CLASS': params['cls_name'][0]})

    try:
        # save trainable class to local file
        if os.path.exists(f'{TEMP_DIR}/ml/') is False:
            os.mkdir(f'{TEMP_DIR}/ml/')
        with open(f'{TEMP_DIR}/ml/{params["module_name"]}.py', 'w') as file:
           file.write(trainable_code)
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