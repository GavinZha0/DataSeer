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
        tags=dict(user_id=user_id, algoId=algo_id, algoName=algo_name, published=False)
    )
    return result.version


"""
un-register a model
"""

async def exper_unreg(algo_id: int, version: int, user_id: int):
    # mlflow.set_tracking_uri(settings.SQLALCHEMY_MLFLOW_DB_URL)
    client = mlflow.MlflowClient(settings.SQLALCHEMY_MLFLOW_DB_URL)

    # md = client.get_model_version(name=f'{algo_id}_{user_id}', version=version)
    # filter_str = f"run_id='{md.run_id}'"
    # mvers = mlflow.search_model_versions(filter_string=filter_str, max_results=1)

    client.set_model_version_tag(f'{algo_id}_{user_id}', version, "published", False)
    client.delete_model_version(name=f'{algo_id}_{user_id}', version=version)


"""
publish a registered model
register the model if it was not registered
"""

async def exper_publish(run_uuid: str, algo_name: str, algo_id: int, user_id: int):
    mlflow.set_tracking_uri(settings.SQLALCHEMY_MLFLOW_DB_URL)

    filter_str = f"run_id='{run_uuid}'"
    mvers = mlflow.search_model_versions(filter_string=filter_str, max_results=1)
    version = None

    if mvers:
        version = mvers[0].version
        # model was registered
        client = mlflow.MlflowClient(settings.SQLALCHEMY_MLFLOW_DB_URL)
        client.set_model_version_tag(f'{algo_id}_{user_id}', version, "published", True)
    else:
        # register and publish
        result = mlflow.register_model(
            model_uri=f'runs:/{run_uuid}/model',
            name=f'{algo_id}_{user_id}',
            tags=dict(user_id=user_id, algoId=algo_id, algoName=algo_name, published=True)
        )
        version = result.version
    return version, True


"""
un-publish a model
"""

async def exper_unpublish(algo_id: int, version: int, user_id: int):
    # mlflow.set_tracking_uri(settings.SQLALCHEMY_MLFLOW_DB_URL)
    client = mlflow.MlflowClient(settings.SQLALCHEMY_MLFLOW_DB_URL)
    client.delete_model_version_tag(name=f'{algo_id}_{user_id}', version=version, key='published')
