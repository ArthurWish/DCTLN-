'''
 * @file 
 * @author CYN <1223174891@qq.com>
 * @createTime 2021/4/9 10:10
'''
import base64
from http.client import HTTPException
from typing import Optional
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import torch.nn as nn
from utils import *
from data import *
from network import TotalNetwork
from easydl import *


class Item(BaseModel):
    # name: str
    # description: Optional[str] = None
    # price: float
    # tax: Optional[float] = None
    signal: str


app = FastAPI()


@app.get('/get_model/')
async def get_model(index: int = 0):
    total_network = TotalNetwork()
    feature_extractor = nn.DataParallel(total_network.feature_extractor.cuda()).train(False)
    classifier = nn.DataParallel(total_network.condition_classifier.cuda()).train(False)
    model = torch.load(os.path.join(train_config.log_dir, 'best_train.pkl'), map_location=torch.device('cpu'))
    feature_extractor.load_state_dict(model['feature_extractor'])
    classifier.load_state_dict(model['condition_class'])
    source_loader = data_loader(data_config.source_speed, data_config.data_type, 'source', is_fft=True)
    data, label = next(iter(source_loader))
    data = data.cuda()
    feature = feature_extractor.forward(data[:index, :])
    _, _, predict = classifier.forward(feature)
    predict = np.argmax(variable_to_numpy(predict), 1)
    return {'predict_label': '{}'.format(predict),
            'ture_label': '{}'.format(label[:index])}

@app.post('/send_data/')
async def send_data(data: Item):
    if not data:
        raise HTTPException(status_code=502, detail="数据发送错误")
    # return data
    r = base64.b64decode(data.signal)
    q = np.frombuffer(r, dtype=np.float32)
    signal = q.reshape(2, 1, 1024)
    signal = torch.from_numpy(signal)
    print(signal)
    total_network = TotalNetwork()
    feature_extractor = nn.DataParallel(total_network.feature_extractor.cuda()).train(False)
    classifier = nn.DataParallel(total_network.condition_classifier.cuda()).train(False)
    model = torch.load(os.path.join(train_config.log_dir, 'best_train.pkl'), map_location=torch.device('cpu'))
    feature_extractor.load_state_dict(model['feature_extractor'])
    classifier.load_state_dict(model['condition_class'])
    feature = feature_extractor.forward(signal)
    _, _, predict_prob = classifier.forward(feature)
    predict_label = np.argmax(variable_to_numpy(predict_prob), 1)
    return {'predict_probability': '{}'.format(predict_prob),
            'predict_label': '{}'.format(predict_label)}


if __name__ == '__main__':
    uvicorn.run(app=app, host="0.0.0.0", port=8000)
