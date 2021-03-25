import os
import random

import numpy as np
import torch
from easydl import select_GPUs
from scipy.fftpack import fft
from scipy.io import loadmat
from torch.utils.data import TensorDataset, DataLoader

dict_fault = {
    '0': '0.007-InnerRace',
    '1': '0.007-OuterRace6',
    '2': '0.014-InnerRace',
    '3': '0.014-OuterRace6',
    '4': '0.021-InnerRace',
    '5': '0.021-OuterRace6',
    '6': 'Normal',
    '7': '0.007-Ball',
    '8': '0.014-Ball',
    '9': '0.021-Ball',
}
fault_class = [dict_fault['6'], dict_fault['1'], dict_fault['0'], dict_fault['7'], dict_fault['4'], dict_fault['5'],
               dict_fault['3'], dict_fault['8'], dict_fault['2'], dict_fault['9'], ]

fault_class_partial = [fault_class[0], fault_class[1], fault_class[2], fault_class[3], fault_class[4], fault_class[5]
                       , fault_class[6]]
# fault_class[8],fault_class[9],], fault_class[3],, fault_class[1],fault_class[2]
#                        fault_class[4],fault_class[5],fault_class[6]

n_total = len(fault_class)  # 总类别数
n_partial = len(fault_class_partial)  # 目标域类别


def add_noise(x, snr):
    """输入原信号，信噪比，输出，加入高斯白噪声的信号
    :param x: 原信号
    :param snr: 信噪比
    :return: 加入高斯白噪声的信号
    """
    P_signal = np.sum(abs(x) ** 2) / len(x)
    P_noise = P_signal / 10 ** (snr / 10.0)
    return x + np.random.randn(len(x)) * np.sqrt(P_noise)


def one_hot_np(class_ids, num_classes):
    """
    创建one-hot格式的标签
    :param class_ids:
    :param num_classes:
    :return:
    """
    labels = torch.tensor(class_ids).view(-1, 1)
    batch_size = labels.numel()
    return torch.zeros(batch_size, num_classes, dtype=torch.float, device=labels.device).scatter_(1, labels, 1).numpy()


class data_config:
    '''
    数据设置类
    '''
    source_speed = 1797
    target_speed = 1730
    data_type = '12k_Drive_End'  # 驱动端
    batch_size = 128
    data_step = 300
    data_length = 1024
    multiple = 1
    add_snr = 2  # 信噪比


class train_config:
    log_dir = './SaveModel/' + 'S' + str(data_config.source_speed) + '_' + str(n_total) + \
              'T_' + str(data_config.target_speed) + '_' + str(n_partial)
    fig_path = './SavePicture/' + 'S' + str(data_config.source_speed) + '_' + str(n_total) + \
               'T_' + str(data_config.target_speed) + '_' + str(n_partial)
    train_step = 50
    train_lr = 0.001
    train_momentum = 0.9
    train_wight_decay = 0.0005
    gpus = 1
    gpu_ids = select_GPUs(gpus)
    output_device = gpu_ids[0]
    is_plot = False
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        print('模型保存地址：', log_dir)
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
        print('图片保存地址：', fig_path)


fault_labels = [i for i in range(0, n_total)]
fault_labels_oh = one_hot_np(fault_labels, n_total)

fault_labels_partial = [i for i in range(0, n_partial)]
fault_labels_partial_oh = one_hot_np(fault_labels_partial, n_partial)


def load_mat(file_path, label, multiple, n_class, snr, number=400, noise=False):
    """
    读取文件中的.mat文件，并预处理
    :param file_path: 读取文件路径
    :param label: 读取文件标签
    :param multiple:
    :param n_class: 共有几类
    :param snr: 信噪比
    :param number: 数据增强
    :param noise: 是否加噪
    :return: 处理后的数据与标签
    """
    i = 0
    data_x = np.zeros((0, data_config.data_length))
    temp = np.zeros((0, data_config.data_length))
    data_y = np.zeros((0, n_class))
    mat_dict = loadmat(file_path)
    print('File_path:{}'.format(file_path))
    filter_i = filter(lambda x: 'DE_time' in x, mat_dict.keys())
    filter_list = [item for item in filter_i]
    key = filter_list[0]
    time_series = mat_dict[key][:, 0]
    step = int(data_config.data_length / multiple)
    start = step * i
    new_time_serices = time_series[start:]

    if noise:
        print('The data is added {}db noise'.format(snr))
        new_time_serices = add_noise(new_time_serices, snr)
    # 数据增强，等步长采样
    if number:
        for k in range(number):
            sample = new_time_serices[k * data_config.data_step: k * data_config.data_step + data_config.data_length]
            temp = np.vstack((temp, sample))
        n = temp.shape[0]
        data_x = temp
        data_y = np.tile(label, (n, 1))
    else:
        idx_last = -(new_time_serices.shape[0] % data_config.data_length)
        clips = new_time_serices[:idx_last].reshape(-1, data_config.data_length)
        n = clips.shape[0]
        data_x = np.vstack((data_x, clips))
        y = np.tile(label, (n, 1))
        data_y = np.vstack((data_y, y))  # data_x:(60, 2048), data_y:(60, 7)
    return data_x, data_y


def load_all_data(speed, data_type):
    """
    :param speed: 读取对应速度
    :param data_type: 数据类别
    :param is_proposed_method: 根据方法，选取读取数据路径
    :return: 数据，标签
    """
    print('***********load all class***********')
    data = np.zeros((0, data_config.data_length))
    label = np.zeros((0, n_total))
    for i in range(n_total):
        print('当前载入数据为：{}, 样本长度为：{}, one-hot标签为：{}， 普通标签为：{}'.format(fault_class[i],
                                                                     data_config.data_length, fault_labels_oh[i],
                                                                     fault_labels[i]))
        x, y = load_mat('CWRU/' + data_type + '/' + str(speed) + '/' +
                        fault_class[i] + '.mat', fault_labels_oh[i], data_config.multiple, n_total,
                        snr=data_config.add_snr, noise=False)
        data = np.vstack((data, x))
        label = np.vstack((label, y))

    # shuffel
    index = list(range(data.shape[0]))
    random.Random(0).shuffle(index)
    data = data[index]
    label = label[index]

    # 数据转换为tensor
    data = torch.from_numpy(data).float()
    data = data.unsqueeze(1)
    label = torch.from_numpy(label).float()

    # 将one hot标签转换为普通标签
    label = torch.topk(label, 1)[1].squeeze(1)

    return data, label


def load_partial_data(speed, data_type):
    """
    :param speed: 读取对应速度
    :param data_type: 数据类别
    :param is_proposed_method: 根据方法，选取读取数据路径
    :return: 数据，标签
    """
    print('***********load partial class***********')
    data = np.zeros((0, data_config.data_length))
    label = np.zeros((0, n_partial))
    for i in range(n_partial):
        print('当前载入数据为：{}, 样本长度为：{}, one-hot标签为：{}, 普通标签为：{}'.format(fault_class_partial[i],
                                                                     data_config.data_length,
                                                                     fault_labels_partial_oh[i],
                                                                     fault_labels_partial[i]))
        x, y = load_mat('CWRU/' + data_type + '/' + str(speed) + '/' +
                        fault_class_partial[i] + '.mat', fault_labels_partial_oh[i], data_config.multiple,
                        n_partial,
                        snr=data_config.add_snr, noise=True)
        data = np.vstack((data, x))
        label = np.vstack((label, y))

    # shuffle
    index = list(range(data.shape[0]))
    random.Random(0).shuffle(index)
    data = data[index]
    label = label[index]

    # 数据转换为tensor
    data = torch.from_numpy(data).float()
    data = data.unsqueeze(1)
    label = torch.from_numpy(label).float()

    # 将one hot标签转换为普通标签
    label = torch.topk(label, 1)[1].squeeze(1)

    return data, label


def data_loader(speed, type, data_from, is_fft=True):
    '''
    数据加载器
    :param speed: 转速
    :param type: 故障类型
    :param data_from: 源域？目标域
    :param is_fft: 是否转频域
    :return: 数据加载器
    '''
    if data_from == 'source':
        source_data, source_label = load_all_data(speed, type)
        if is_fft:
            print('[INFORMATION]:*****The data is use FFT\n')
            source_data = torch.FloatTensor(np.abs(fft(source_data.numpy())))
        len_s = int(source_data.shape[0] * 0.8)
        source_data_set = TensorDataset(source_data[:len_s], source_label[:len_s])
        data_loader = DataLoader(dataset=source_data_set, batch_size=data_config.batch_size, shuffle=True)
    if data_from == 'target':
        target_data, target_label = load_partial_data(speed, type)
        if is_fft:
            target_data = torch.FloatTensor(np.abs(fft(target_data.numpy())))
        len_t = int(target_data.shape[0] * 0.8)
        target_data_set = TensorDataset(target_data[:len_t], target_label[:len_t])
        data_loader = DataLoader(target_data_set, data_config.batch_size, shuffle=True)

    return data_loader
