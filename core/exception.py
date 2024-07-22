# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2021/10/19
# @File           : exception.py
# @desc           : Global exception

from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.exceptions import RequestValidationError
from starlette import status
from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI
from core.logger import logger
from config.settings import DEVELOPMENT


class CustomException(Exception):

    def __init__(
            self,
            msg: str,
            code: int = status.HTTP_400_BAD_REQUEST,
            status_code: int = status.HTTP_200_OK,
            desc: str = None
    ):
        self.msg = msg
        self.code = code
        self.status_code = status_code
        self.desc = desc


def register_exception(app: FastAPI):
    @app.exception_handler(CustomException)
    async def custom_exception_handler(request: Request, exc: CustomException):
        if DEVELOPMENT:
            print("URL", request.url.__str__())
            print("CustomException：custom_exception_handler")
            print(exc.desc)
            print(exc.msg)
        logger.exception(exc)
        return JSONResponse(
            status_code=exc.status_code,
            content={"msg": exc.msg, "code": exc.code},
        )

    @app.exception_handler(StarletteHTTPException)
    async def unicorn_exception_handler(request: Request, exc: StarletteHTTPException):
        if DEVELOPMENT:
            print("HTTPException:", request.url.__str__())
            print(exc.detail)
        logger.exception(exc)
        return JSONResponse(
            status_code=200,
            content={
                "code": exc.status_code,
                "message": exc.detail,
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        if DEVELOPMENT:
            print("URL", request.url.__str__())
            print("CustomException：validation_exception_handler")
            print(exc.errors())
        logger.exception(exc)
        msg = exc.errors()[0].get("msg")
        return JSONResponse(
            status_code=200,
            content=jsonable_encoder(
                {
                    "msg": msg,
                    "body": exc.body,
                    "code": status.HTTP_400_BAD_REQUEST
                }
            ),
        )

    @app.exception_handler(ValueError)
    async def value_exception_handler(request: Request, exc: ValueError):
        if DEVELOPMENT:
            print("URL", request.url.__str__())
            print("CustomException：value_exception_handler")
            print(exc.__str__())
        logger.exception(exc)
        return JSONResponse(
            status_code=200,
            content=jsonable_encoder(
                {
                    "msg": exc.__str__(),
                    "code": status.HTTP_400_BAD_REQUEST
                }
            ),
        )

    @app.exception_handler(Exception)
    async def all_exception_handler(request: Request, exc: Exception):
        if DEVELOPMENT:
            print("URL", request.url.__str__())
            print("CustomException：all_exception_handler")
            print(exc.__str__())
        logger.exception(exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=jsonable_encoder(
                {
                    "msg": "interface exception！",
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR
                }
            ),
        )
