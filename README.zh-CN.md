<div align="center"> <a href="https://github.com/GavinZha0/DataPie"> <img alt="DataPie Logo" width="200" height="200" src="/static/system/logo.png"> </a> <br> <br>

<h1>DataPie</h1>
</div>

**中文** | [English](./README.md)

## 简介
DataPie是一个低代码数据平台，为快速构建AI和BI应用而生。
- DataExplorer: DataPie前端 (Js, Ts, Vue, Antd, G2plot, Leaflet, Cy...)
- DataMagic: Java后端服务器 (Springboot, Jwt, Jpa, Druid, Knife4j, Tablesaw, DJL, DL4J...)
- DataSeer: Python后端服务器 (FastApi, Sk-learn, Pytorch, MlFlow, Ray... )

## 特性
- **数据可视化**：很容易的连结数据库，生成视图并构建报表！(原型已就绪)
- **机器学习**: 快速地开发算法，构建模型并训练模型！(2024年完成)
- **人工智能**：快捷地管理预训练模型及生成AI应用！(2024年完成)

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
