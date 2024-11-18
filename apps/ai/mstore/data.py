import requests

async def data_execute(url: str, data: dict):
    result = requests.post(url=url, json=data)
    result.encoding = 'utf-8'
    return result.json()