'''
 * @file 
 * @author CYN <1223174891@qq.com>
 * @createTime 2021/4/9 13:15
'''
import base64

import fastapi
import requests
from starlette.testclient import TestClient

from data import data_loader, data_config

app = fastapi.FastAPI()
get_url = 'http://cyn.apustech.cn:8000/get_model'
post_url = 'http://cyn.apustech.cn:8000/send_data'
client = TestClient(app)
# get_url = 'http://localhost:8000/get_model'
# post_url = 'http://localhost:8000/send_data'

get_param = {
    "index": 10,
}
source_loader = data_loader(data_config.source_speed, data_config.data_type, 'source', is_fft=True)
data, label = next(iter(source_loader))
data = data[:2]
t = data.numpy()
print(t.dtype, t)
s: bytes = base64.b64encode(t)
post_param = {
    "signal": s.decode("utf-8")
}

# 这里是去请求后端
def get(url, param):
    data = requests.get(url, param)
    print(data)
    y = data.json()
    print(y)
    return y


def post(url, param):
    data = requests.post(url, json=param)
    print(data.json())
    return data


if __name__ == '__main__':
    # get(get_url, get_param)
    post(post_url, post_param)
