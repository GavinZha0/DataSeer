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
