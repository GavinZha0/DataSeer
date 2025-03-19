# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2021/10/19
# @File           : main.py
# @desc           : main
import sys
from fastapi import FastAPI
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from config import settings, router
from starlette.staticfiles import StaticFiles
from config.settings import REDIS_ENABLE, RAY_NUM_CPU, RAY_NUM_GPU, RAY_LOCAL_MODE, TEMP_DIR, RAY_ENABLE
from core.docs import custom_api_docs
from core.exception import register_exception
import typer
from fastapi.middleware.gzip import GZipMiddleware
from msgq.redis_listener import RedisListener
from core.event import lifespan
from utils.tools import import_modules

shell_app = typer.Typer()


def create_app():
    """
    create FastAPI app
    docs_url： /docs
    redoc_url: /redoc
    """
    app = FastAPI(
        title=settings.TITLE,
        description=settings.DESCRIPTION,
        version=settings.VERSION,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None
    )
    import_modules(settings.MIDDLEWARES, "Middlewares", app=app)
    register_exception(app)

    # support zip in response
    if settings.HTTP_RESPONSE_ZIP:
        app.add_middleware(GZipMiddleware, minimum_size=1000)

    # cross domain
    if settings.CORS_ORIGIN_ENABLE:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.ALLOW_ORIGINS,
            allow_credentials=settings.ALLOW_CREDENTIALS,
            allow_methods=settings.ALLOW_METHODS,
            allow_headers=settings.ALLOW_HEADERS
        )
    # static folder
    if settings.STATIC_ENABLE:
        app.mount(settings.STATIC_URL, app=StaticFiles(directory=settings.STATIC_ROOT))

    # app router
    for route in router.urlpatterns:
        app.include_router(route["ApiRouter"], prefix=route["prefix"], tags=route["tags"])

    # doc api
    custom_api_docs(app)
    return app


@shell_app.command()
def run(
        host: str = typer.Option(default='0.0.0.0', help='Host ip'),
        port: int = typer.Option(default=9138, help='Port')
):
    """
    start application
    """
    if REDIS_ENABLE:
        RedisListener()

    if RAY_ENABLE:
        import ray
        # init ray to connect to cluster when cluster mode
        # give a temp folder for ray using
        print('initialize RAY.......')
        ray.init(ignore_reinit_error=True, local_mode=RAY_LOCAL_MODE,
                 num_cpus=RAY_NUM_CPU, num_gpus=RAY_NUM_GPU, _temp_dir=TEMP_DIR+'/ray/')
        # ray.autoscaler.sdk.request_resources(bundles=[{"GPU": 1}] * 1)

    # start python server with single worker or multiple workers
    # different workers can be run on different CPU cores
    print('starting app.......')
    uvicorn.run(app='main:create_app', host=host, port=port, workers=1, lifespan="on", factory=True)
    print('exiting app......')


@shell_app.command()
def init():
    print("initialize...")

if __name__ == '__main__':
    shell_app()