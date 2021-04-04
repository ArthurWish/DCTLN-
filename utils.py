'''
 * @file 一些工具函数和类可以放这里
 * @author CYN <1223174891@qq.com>
 * @createTime 2021/3/24 10:41
'''
import logging
import yaml
import logging.config
from matplotlib import pyplot as plt
from easydl import AccuracyCounter, variable_to_numpy, TrainingModeManager, one_hot
from sklearn.manifold import TSNE

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

def plot_tsne3(data1, label1, data2, label2, n_source, n_target, epoch, acc, fig_path, title):
    """绘制tsne可视化
        :param data1: 源域
        :param data2: 目标域
        """
    plt.figure(figsize=(8, 6))
    plt.xticks([]), plt.yticks([])
    # plt.xlim([min(test_fc2_tsne[:, 0] - 5), max(test_fc2_tsne[:, 0]) + 5])  # 可视化前设置坐标系的取值范围
    # plt.ylim([min(test_fc2_tsne[:, 1] - 5), max(test_fc2_tsne[:, 1]) + 5])
    colors = ['red', 'orange', 'yellow', 'green', 'cyan',
              'blue', 'purple', 'pink', 'magenta', 'brown']
    data1 = data1.cpu().numpy()
    data2 = data2.cpu().numpy()
    lenth1 = data1.shape[0]
    data_compact = np.vstack((data1, data2))

    print('Plotting t-sne-version3.0 figure...'+epoch)
    # 源域目标域一起进行tsne降维
    tsne = TSNE(n_components=2, init='pca', random_state=0, n_iter=500)
    data_compact = tsne.fit_transform(data_compact)
    data1_tsne = data_compact[:lenth1, :]
    data2_tsne = data_compact[lenth1:, :]
    # 归一化
    x_min, x_max = np.min(data1_tsne), np.max(data1_tsne)
    data1_tsne = (data1_tsne - x_min) / (x_max - x_min) * 10
    data2_tsne = (data2_tsne - x_min) / (x_max - x_min) * 10
    for c1 in range(n_source):
        indices = np.where(label1 == c1)
        indices = indices[0]
        plt.scatter(data1_tsne[indices, 0], data1_tsne[indices, 1], marker='o', label=c1, c=colors[c1], s=30, alpha=0.9)

    for c1 in range(n_target):
        indices = np.where(label2 == c1)
        indices = indices[0]
        plt.scatter(data2_tsne[indices, 0], data2_tsne[indices, 1], marker='X', label=c1, c=colors[c1], s=30, alpha=0.9)

    plt.legend(loc=3, bbox_to_anchor=(1.005, 0), borderaxespad=0)
    plt.title(title+'T-SNE-{}-Acc={:.4f}'.format(epoch, acc))
    # plt.savefig('./Figure/{}-Acc={:.2f}.png'.format(epoch, acc))
    plt.savefig(fig_path + '/{}-Acc={:.4f}.png'.format(epoch, acc))