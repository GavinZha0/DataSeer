import typing
from fastapi.responses import ORJSONResponse as Response
from fastapi import status as http_status
from utils import status as http


class SuccessResponse(Response):
    def __init__(self, data=None, msg="success", code=http.HTTP_SUCCESS, status=http_status.HTTP_200_OK, headers: typing.Mapping[str, str] | None = None
                 , **kwargs):
        self.data = {
            "code": code,
            "msg": msg,
            "data": data
        }
        self.data.update(kwargs)
        super().__init__(content=self.data, status_code=status, headers=headers)


class ErrorResponse(Response):
    def __init__(self, msg=None, code=http.HTTP_ERROR, status=http_status.HTTP_200_OK, **kwargs):
        self.data = {
            "code": code,
            "msg": msg,
            "data": []
        }
        self.data.update(kwargs)
        super().__init__(content=self.data, status_code=status)
