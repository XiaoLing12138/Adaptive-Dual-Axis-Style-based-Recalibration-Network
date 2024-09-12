import os
import csv
import random
import shutil

import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.models as mds

from data.dataloaders.dataloaders_isic import ISICDataloader
import yaml
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from torch.autograd import Variable
import torch.nn.functional as F

from utils import *

with open('config_loss.yaml', 'rb') as f:
    config = yaml.safe_load(f.read())

image_size = config["train"]["size"]
resume = config["train"]["resume"]
random_seed = config["train"]["seed"]
basic_learning_rate = config["train"]["lr"]
batch_size = config["train"]["batch-size"]
image_number = config["train"]["img_num"]
epochs = config["train"]["epochs"]
decay = config["train"]["decay"]
dataset = config["train"]["dataset"]
num_worker = config["train"]["num_worker"]
data_path = config["train"]["data_path"]
num_class = config["train"]["num_classes"]
bool_pretrain = config["train"]["pretrain"]

mixed_loss_bool = config["train"]["mixed_loss"]
loss_file = config["model"]["loss_file"]
loss_caller = config["model"]["loss_caller"]
loss_caller_mix = config["model"]["loss_caller_mix"]
loss_alpha = config["model"]["alpha"]
loss_beta = config["model"]["beta"]

caller = config["model"]["caller"]
model_name = config["model"]["name"]
file_name = config["model"]["file"]
checkpoint_folder = config["model"]["folder"]
model_pretrain = config["model"]["pretrain"]

log_folder = config["log"]["folder"]

exec('from losses.{} import *'.format(loss_file))
exec('from models.{} import *'.format(file_name))

torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

best_acc = 0

lws = None

os.environ["CUDA_VISIBLE_DEVICES"] = str(config['train']['cuda'])
use_cuda = torch.cuda.is_available()

if not os.path.isdir(checkpoint_folder):
    os.mkdir(checkpoint_folder)

if not os.path.isdir(log_folder):
    os.mkdir(log_folder)

shutil.copyfile(os.path.join(sys.path[0], 'config_loss.yaml'), os.path.join(sys.path[0], log_folder,
                                                                            model_name + '_' + str(random_seed) +
                                                                            '_config.log'))

log_file = os.path.join(sys.path[0], log_folder, model_name + '_' + str(random_seed) + '.csv')
if not os.path.exists(log_file):
    with open(log_file, 'w') as f:
        log_writer = csv.writer(f, delimiter=',')
        log_writer.writerow(['epoch', 'train loss', 'train acc', 'test loss', 'test acc'])


def save_checkpoint(model, acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': model,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }

    torch.save(state, os.path.join(sys.path[0], checkpoint_folder,
                                   model_name + '_' + str(random_seed) + '.pth'))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = basic_learning_rate
    if epoch <= 9:
        # warm-up training for large minibatch
        lr = basic_learning_rate + basic_learning_rate * epoch / 10.
    if epoch >= 100 and epoch % 10 == 0:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, train_loader, epoch, criterion, optimizer):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_index = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_index = batch_idx

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)
        if loss_caller == "LearnableWeightScalingLoss":
            outputs = lws(outputs)

        if mixed_loss_bool:
            loss1 = criterion[0](outputs, targets)
            loss2 = criterion[1](outputs, targets)
            loss = loss_alpha * loss1 + loss_beta * loss2
        else:
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct = correct.item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return train_loss / batch_index, 100. * correct / total


def valid(model, valid_loader, valid_loader_train, epoch, criterion):
    global best_acc
    model.eval()
    valid_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            if loss_caller == "LearnableWeightScalingLoss":
                outputs = lws(outputs)

            if mixed_loss_bool:
                loss1 = criterion[0](outputs, targets)
                loss2 = criterion[1](outputs, targets)
                loss = loss_alpha * loss1 + loss_beta * loss2
            else:
                loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

            progress_bar(batch_idx, len(valid_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (valid_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        for batch_idx, (inputs, targets) in enumerate(valid_loader_train):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            if loss_caller == "LearnableWeightScalingLoss":
                outputs = lws(outputs)

            if mixed_loss_bool:
                loss1 = criterion[0](outputs, targets)
                loss2 = criterion[1](outputs, targets)
                loss = loss_alpha * loss1 + loss_beta * loss2
            else:
                loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

            progress_bar(batch_idx, len(valid_loader_train), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (valid_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        save_checkpoint(model, acc, epoch)
    return valid_loss / batch_idx, 100. * correct / total


def classify_report(result):
    result = result.split()
    x = result.index('accuracy')
    n = (x + 1) // 5 - 1
    tag = result[:4]
    m = {}
    for i in range(n):
        start = result.index(str(i)) + 1
        for j in range(4):
            m["grade" + str(i) + "_" + tag[j]] = result[start + j]
    m[result[x]] = result[x + 1]
    m_x = result.index("macro")
    for i in range(3):
        m[result[m_x] + "_" + tag[i]] = result[m_x + 2 + i]
    w_x = result.index("weighted")
    for i in range(3):
        m[result[w_x] + "_" + tag[i]] = result[w_x + 2 + i]
    m["total"] = result[-1]
    return m


def assemble_labels(y_true, y_pred, label, out):
    if y_true == None:
        y_true = label
        y_pred = out
    else:
        y_true = torch.cat((y_true, label), 0)
        y_pred = torch.cat((y_pred, out))

    return y_true, y_pred


def get_ece(logits, labels, n_bins=15):
    # This function is based on https://github.com/gpleiss/temperature_scaling
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=logits.device)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    # bin_boundaries, bin_lowers, bin_uppers

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()  # weight of current bin

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def get_mce(logits, labels, n_bins=15):
    # This function is based on https://github.com/gpleiss/temperature_scaling
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    mce = torch.zeros(1, device=logits.device)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    # bin_boundaries, bin_lowers, bin_uppers

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            error = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
            if (error > mce):
                mce = error

    return mce


def get_bs(logits, labels):
    softmaxes = F.softmax(logits, dim=1)
    bs_score = 0.0
    for i in range(len(labels)):
        for j in range(len(softmaxes[0])):
            if labels[i] == j:
                bs_score += (softmaxes[i][j] - 1) ** 2
            else:
                bs_score += (softmaxes[i][j] - 0) ** 2

    bs_score /= len(labels)
    return bs_score


# Testing
def test(test_loader, criterion):
    print("begin")
    global best_acc
    model_path = os.path.join(sys.path[0], checkpoint_folder, model_name + '_' + str(random_seed) + '.pth')
    checkpoint = torch.load(model_path)
    model = checkpoint['net']
    best_acc = checkpoint['acc']
    epoch = checkpoint['epoch']
    torch.set_rng_state(checkpoint['rng_state'])

    if use_cuda:
        model.cuda()
        print('Using', torch.cuda.device_count(), 'GPUs.')
        cudnn.deterministic = True
        print('Using CUDA..')
    model.eval()
    print(epoch)
    valid_loss = 0
    correct = 0
    total = 0
    y_true = None
    y_pred = None
    labels = []
    pre_labels = []
    auc_output = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            if loss_caller == "LearnableWeightScalingLoss":
                outputs = lws(outputs)
            auc_output.extend(outputs.data.cpu().numpy())

            if mixed_loss_bool:
                loss1 = criterion[0](outputs, targets)
                loss2 = criterion[1](outputs, targets)
                loss = loss_alpha * loss1 + loss_beta * loss2
            else:
                loss = criterion(outputs, targets)

            y_true, y_pred = assemble_labels(y_true, y_pred, targets, outputs)

            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            labels.extend(targets.data.cpu())
            pre_labels.extend(predicted.data.cpu())
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (valid_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    ece = get_ece(y_pred, y_true)
    mce = get_mce(y_pred, y_true)
    bs = get_bs(y_pred, y_true)

    labels = np.reshape(labels, [-1, 1])
    pre_labels = np.reshape(pre_labels, [-1, 1])
    print(classification_report(labels, pre_labels, digits=5))
    print("balanced_acc: ")
    print(balanced_accuracy_score(labels, pre_labels))
    # label -->
    auc_labels = np.zeros(shape=(len(labels), config["train"]["num_classes"]))
    for i in range(len(labels)):
        auc_labels[i][labels[i]] = 1

    auc_output = np.array(auc_output)

    auc_macro_ovo = roc_auc_score(auc_labels, auc_output, average='macro', multi_class='ovo')
    auc_macro_ovr = roc_auc_score(auc_labels, auc_output, average='macro', multi_class='ovr')
    auc_weighted_ovo = roc_auc_score(auc_labels, auc_output, average='weighted', multi_class='ovo')
    auc_weighted_ovr = roc_auc_score(auc_labels, auc_output, average='weighted', multi_class='ovr')

    with open(log_file, 'a') as f:
        f.write(classification_report(labels, pre_labels, digits=5))
        f.write("epoch: ")
        f.write(str(epoch))
        f.write("\n")
        f.write("valid_acc: ")
        f.write(str(best_acc))
        f.write("\n")
        f.write("kappa: ")
        f.write(str(cohen_kappa_score(labels, pre_labels)))
        f.write("\n")
        f.write("auc_macro_ovo: ")
        f.write(str(auc_macro_ovo))
        f.write("\n")
        f.write("auc_macro_ovr: ")
        f.write(str(auc_macro_ovr))
        f.write("\n")
        f.write("auc_weighted_ovo: ")
        f.write(str(auc_weighted_ovo))
        f.write("\n")
        f.write("auc_weighted_ovr: ")
        f.write(str(auc_weighted_ovr))
        f.write("\n")
        f.write("balanced_acc: ")
        f.write(str(balanced_accuracy_score(labels, pre_labels)))
        f.write("\n")
        f.write("ece: ")
        f.write(str(ece))
        f.write("\n")
        f.write("mce: ")
        f.write(str(mce))
        f.write("\n")
        f.write("bs: ")
        f.write(str(bs))


def main():
    global best_acc, dataset, lws

    dataloader = ISICDataloader(
        batch_size=batch_size,
        num_workers=num_worker,
        img_resize=image_size,
        root_dir=data_path,
    )

    (test_loader, test_dataset,) = dataloader.run("test")

    (train_loader, train_dataset,) = dataloader.run("train")

    # model initial
    model = eval('{}(num_classes=num_class)'.format(caller))
    if bool_pretrain:
        premodel_dict = eval('mds.{}(weights="IMAGENET1K_V1")'.format(model_pretrain))
        pretrain_dict = premodel_dict.state_dict()
        model_dict = model.state_dict()
        select_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and k[:2] != 'fc'}
        model_dict.update(select_dict)
        model.load_state_dict(model_dict)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.deterministic = True
    print('Using', torch.cuda.device_count(), 'GPUs.')
    print('Using CUDA..')

    if mixed_loss_bool:
        criterion = [eval('{}()'.format(loss_caller)), eval('{}()'.format(loss_caller_mix))]
    else:
        if loss_caller == "TemperatureScaling":
            criterion = nn.CrossEntropyLoss()
        elif loss_caller == "LearnableWeightScalingLoss":
            criterion = eval("LabelAwareSmoothing()")
            lws = eval('{}()'.format(loss_caller))
            lws = torch.nn.DataParallel(lws).cuda()
        else:
            criterion = eval('{}()'.format(loss_caller))

    if loss_caller == "LearnableWeightScalingLoss":
        optimizer = optim.SGD([{"params": model.parameters()},
                               {'params': lws.parameters()}], lr=basic_learning_rate, momentum=0.9, weight_decay=decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=basic_learning_rate, momentum=0.9, weight_decay=decay)

    if resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(os.path.join(sys.path[0], checkpoint_folder)), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join(sys.path[0], checkpoint_folder,
                                             model_name + '_' + str(random_seed) + '.pth'))
        model = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
    else:
        print('==> Building model...')
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, epoch)
        # complement_adjust_learning_rate(complement_optimizer, epoch)
        train_loss, train_acc = train(model, train_loader, epoch, criterion, optimizer)
        valid_loss, valid_acc = valid(model, test_loader, test_loader, epoch, criterion)
        with open(log_file, 'a') as f:
            log_writer = csv.writer(f, delimiter=',')
            log_writer.writerow([epoch, train_loss, train_acc, valid_loss, valid_acc])

    test(test_loader, criterion)


if __name__ == '__main__':
    main()
