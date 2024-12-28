#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/7/13
# @File           : db_executor.py
# @desc           : DB operation

import pandas as pd
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from core.logger import logger

# database driver should be installed if the error 'No module named MySQLdb' is shown
# the url should be 'mysql+mysqldb' when mysqlclient is used
# the url should be 'mysql+pymysql' when pymysql is used. And pymysql.install_as_MySQLdb() should be added to class init

class DbExecutor:

    def __init__(self, type: str = None, url: str = None, passport: str = None, params: str = None):
        self.engine = None
        self.connection = None

        # enable it when pymysql is used as database driver to execute original sql statement
        # pymysql.install_as_MySQLdb()

        if type != None:
            # crate async engine
            self.engine = create_async_engine(f'mysql+asyncmy://{passport}@{url}?charset=utf8mb4', echo=False)

    async def db_query(self, sql: str = None, limit: int = None, params: dict = None) -> None:
        try:
            # get data from db synchronously
            async with self.engine.connect() as conn:
                result = await conn.execute(text(sql))
            col_names = result.keys()
            data = result.fetchall()
            df = pd.DataFrame.from_records(data, columns=col_names)

            total = len(df)
            if limit:
                return df[0:limit], total
            else:
                return df, total
        except AttributeError as e:
            logger.error(f"Failed to parse db url，{e}")
            raise ValueError("Failed to parse db url！")
            return None, None

    def db_close(self) -> None:
        try:
            self.connection.close()
            self.engine.dispose()
        except AttributeError as e:
            logger.warning(f"Close unconnected db！")


if __name__ == '__main__':
    t = DbExecutor()
    t.db_query()
    t.db_close()
