# Deep Convolutional Transfer Learning Network
from torch import optim
from tqdm import trange
from network import *
from utils import *

# initial logger
setup_logging(default_path="train_log.yaml")
logging.info('\n****[源域]****\n[转速]:{}\n[类数]:{}\n[类别]:{}\n'
             '****[目标域]****\n[转速]:{}\n[类数]:{}\n[类别]:{}\n'
             '[训练轮次]: {}\n[学习率]: {}\n[噪声]: {}dB'
             .format(data_config.source_speed, n_total, fault_class,
                     data_config.target_speed, n_partial, fault_class_partial,
                     train_config.train_step, train_config.train_lr, data_config.add_snr))

# initial network
total_network = TotalNetwork()
feature_extractor = nn.DataParallel(total_network.feature_extractor,
                                    device_ids=train_config.gpu_ids, output_device=train_config.output_device).train(
    True)
condition_classifier = nn.DataParallel(total_network.condition_classifier,
                                       device_ids=train_config.gpu_ids, output_device=train_config.output_device).train(
    True)
domain_classifier = nn.DataParallel(total_network.domain_classifier,
                                    device_ids=train_config.gpu_ids, output_device=train_config.output_device).train(
    True)
# Set optimizer
scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000)
optimizer_extractor = OptimWithSheduler(
    optim.SGD(feature_extractor.parameters(), lr=train_config.train_lr, weight_decay=train_config.train_wight_decay,
              momentum=train_config.train_momentum, nesterov=True), scheduler)
optimizer_con_cls = OptimWithSheduler(
    optim.SGD(condition_classifier.parameters(), lr=train_config.train_lr, weight_decay=train_config.train_wight_decay,
              momentum=train_config.train_momentum, nesterov=True), scheduler)
optimizer_domain_cls = OptimWithSheduler(
    optim.SGD(domain_classifier.parameters(), lr=train_config.train_lr, weight_decay=train_config.train_wight_decay,
              momentum=train_config.train_momentum, nesterov=True), scheduler)
# Load data
source_loader = data_loader(data_config.source_speed, data_config.data_type, 'source', is_fft=True)
target_loader = data_loader(data_config.target_speed, data_config.data_type, 'target', is_fft=True)

# Train network
best_source_acc, best_target_acc = 0, 0
list_acc_s, list_acc_t = [], []
CrossEntropyLoss = nn.CrossEntropyLoss()
BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
MMDLoss = MMD_loss()

for epoch in trange(train_config.train_step, desc='DCTLN epoch', unit='epoch'):
    i = 0
    for (source, source_label), (target, _) in zip(source_loader, target_loader):
        i += 1
        source, source_label = source.to(train_config.output_device), source_label.to(train_config.output_device)
        target = target.to(train_config.output_device)
        # forward pass
        feature_s = feature_extractor.forward(source)  # Calculate source feature
        feature_t = feature_extractor.forward(target)  # Calculate target feature
        _, before_softmax, predict = condition_classifier.forward(feature_s)  # Calculate source class probability
        domain_prob_s = domain_classifier.forward(feature_s)
        domain_prob_t = domain_classifier.forward(feature_t)
        # compute loss
        ce_loss = CrossEntropyLoss(before_softmax,
                                   source_label)  # combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss`
        adv_loss = BCEWithLogitsLoss(domain_prob_s,
                                     torch.ones_like(domain_prob_s))  # combines a `Sigmoid` layer and the `BCELoss`
        adv_loss += BCEWithLogitsLoss(domain_prob_t, torch.zeros_like(domain_prob_t))
        mmd_loss = MMDLoss(feature_s, feature_t)
        # Backward to update the network's parameters
        with OptimizerManager([optimizer_extractor, optimizer_con_cls, optimizer_domain_cls]):
            total_loss = ce_loss + adv_loss + mmd_loss
            total_loss.backward()

        # Calculate Accuracy
        source_acc = calculate_accuracy(feature_extractor=feature_extractor, classifier=condition_classifier,
                                        loader=source_loader, n_total=n_total)
        target_acc = calculate_accuracy(feature_extractor=feature_extractor, classifier=condition_classifier,
                                        loader=target_loader, n_total=n_total)
        print('acc_s:   {}\nacc_t:   {}\n'.format(source_acc, target_acc))
        # Save Accuracy in list to plot them by using matplotlib
        list_acc_s.append(source_acc)
        list_acc_t.append(target_acc)
        # Save Model
        data = {
            'feature_extractor': feature_extractor.state_dict(),
            'condition_class': condition_classifier.state_dict(),
            'domain_class': domain_classifier.state_dict()
        }
        if target_acc > best_target_acc:
            best_target_acc = target_acc
            with open(os.path.join(train_config.log_dir, 'best_train.pkl'), 'wb') as f:
                torch.save(data, f)
            print('[information]*********the model saved**********')

# plot accuracy figure
plt.plot(list_acc_s, 'r-.', label='source')
plt.plot(list_acc_t, 'b--', label='target')
plt.title('DCTLN network')
plt.legend(), plt.savefig(train_config.fig_path + '/Acc={:.4f}.png'.format(best_target_acc)), plt.show()
