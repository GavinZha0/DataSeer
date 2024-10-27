#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/7/13
# @File           : db_executor.py
# @desc           : DB operation

import pandas as pd
import pymysql
from sqlalchemy import create_engine
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
            # driver: mysqlclient (pip install mysqlclient)

            # create sync engine
            # self.engine = create_engine(
            #     'mysql+mysqldb://' + passport + '@' + url + '?charset=utf8mb4',
            #     echo=False)

            # crate async engine
            self.engine = create_async_engine(f'mysql+asyncmy://{passport}@{url}?charset=utf8mb4', echo=False)

            # driver: pymysql
            # self.engine = create_engine('mysql+pymysql://admin:admin520@datapie.cnqbtlcpe5hy.us-east-2.rds.amazonaws.com:3306/foodmart?charset=utf8mb4', echo=False)
            # self.connection = self.engine.connect().execution_options(stream_results=True)

    async def db_query(self, sql: str = None, limit: int = None, params: dict = None) -> None:
        try:
            # query = text("SELECT x, y FROM some_table WHERE y > :y")
            # params = {"y": 2}
            # result_set = self.connection.execute(query, params)

            # query = text(sql)
            # result_set = self.connection.execute(query)
            # for row in result_set:
            #    print(row)

            # read data from db asynchronously
            #df = pd.read_sql(sql, self.engine)

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
        except pymysql.err.OperationalError as e:
            logger.error(f"Failed to connect to db，{e}")
            raise ValueError("Failed to connect to db！")
            return None, None
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
