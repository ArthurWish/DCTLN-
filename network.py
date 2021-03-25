from easydl import *

'''
特征提取器
'''


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Sequential(  # input: 64,1,2048--64-16
            nn.Conv1d(1, 16, kernel_size=64, stride=1, padding=32),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(  # input: 64,16,62
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(  # input: 64,32,31
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 64, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.__in_features = 1024

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(-1, 128 * 64)
        x_fc1 = self.fc1(x)
        return x_fc1

    def output_num(self):
        return self.__in_features


class ConditionClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConditionClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
        )
        self.main = nn.Sequential(self.fc, nn.Softmax(dim=-1))  # 分别返回before softmax 和 softmax的数据

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out


class DomainClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            # nn.BatchNorm1d(out_dim),
            # nn.ReLU(),
            nn.Linear(out_dim, 1),
        )
        # grl 是梯度反转层，用于对抗训练
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        x = self.grl(x)
        output = self.fc1(x)
        return output


class TotalNetwork(nn.Module):
    def __init__(self, n_total=10):
        super(TotalNetwork, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.condition_classifier = ConditionClassifier(self.feature_extractor.output_num(), n_total)
        self.domain_classifier = DomainClassifier(self.feature_extractor.output_num(), out_dim=512)

    def forward(self, x):
        f = self.feature_extractor(x)  # 特征提取器输出
        d = self.domain_classifier(f)  # 域判别器输出
        _, _, y = self.condition_classifier(f)  # 分类器输出
        return f, d, y


# distribution discrepancy: 计算源域和目标域提取特征的分布差异
class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            torch.cuda.empty_cache()
            return loss