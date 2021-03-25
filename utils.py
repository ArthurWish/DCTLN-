'''
 * @file 一些工具函数和类可以放这里
 * @author CYN <1223174891@qq.com>
 * @createTime 2021/3/24 10:41
'''
import logging
import yaml
import logging.config

from easydl import AccuracyCounter, variable_to_numpy, TrainingModeManager, one_hot

from data import *


def setup_logging(default_path="logging.yaml", default_level=logging.INFO, env_key="LOG_CFG"):
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

# 计算模型的分类正确率
def calculate_accuracy(feature_extractor, classifier, loader, n_total):
    counter = AccuracyCounter()
    with TrainingModeManager([feature_extractor, classifier], train=False) as mgr, torch.no_grad():
        for (data, label) in loader:
            data, label = data.cuda(), label.cuda()
            feature = feature_extractor.forward(data)
            _, before_softmax, predict_prob = classifier.forward(feature)
            counter.addOneBatch(variable_to_numpy(predict_prob),
                                variable_to_numpy(one_hot(label, n_total)))
    accuracy = counter.reportAccuracy()
    return accuracy