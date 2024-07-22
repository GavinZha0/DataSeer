<div align="center"> <a href="https://github.com/GavinZha0/DataPie"> <img alt="DataPie Logo" width="200" height="200" src="/static/system/logo.png"> </a> <br> <br>

<h1>DataPie</h1>
</div>

**English** | [中文](./README.zh-CN.md)

## Introduction
DataPie, a low-code data platform for AI and BI.
- DataExplorer: Front end of DataPie (Js, Ts, Vue, Antd, G2plot, Leaflet, Cy...)
- DataMagic: Java server of DataPie (Springboot, Jwt, Jpa, Druid, Knife4j, Tablesaw, DJL, DL4J...)
- DataSeer: Python server of DataPie (FastApi, Sk-learn, Pytorch, MlFlow, Ray... )
## Feature

- **Data visualization**：Connect datasource, build views and publish dashboard easily! (prototype is ready)
- **Machine learning**: Develop algorithm, build model and train ML quickly! (to be done in 2024)
- **AI application building**: Manage model and make application rapidly! (to be done in 2024)

# DataSeer
Backend server of DataPie for ML and AI

## Key components

Python: 3.10

Fastapi: 0.111

SQLAlchemy: 2.0

Plotly: 5.22

Pandas: 2.22

Sklearn: 1.42

Featuretools: 1.31

mlflow: 2.14

ray: 2.32

## usage

```shell
# install components
poetry

# initialize database
python main.py init --env dev

# run application
python main.py run
```


## Other
```shell
# migrate database
python main.py migrate --env dev
```
