import mlflow
from config import settings


"""
register a model with unique name
version will be increased if name exist
"""

async def exper_reg(run_uuid: str, algo_name: str, algo_id: int, user_id: int):
    mlflow.set_tracking_uri(settings.SQLALCHEMY_MLFLOW_DB_URL)
    result = mlflow.register_model(
        model_uri=f'runs:/{run_uuid}/model',
        name=f'{algo_id}_{user_id}',
        tags=dict(user_id=user_id, algoId=algo_id, algoName=algo_name)
    )
    return result.version


"""
un-register a model
"""

async def exper_unreg(algo_id: int, version: int, user_id: int):
    mlflow.set_tracking_uri(settings.SQLALCHEMY_MLFLOW_DB_URL)
    client = mlflow.MlflowClient()
    client.delete_model_version(name=f'{algo_id}_{user_id}', version=version)
