from turtle import back
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class Dataset(Dataset):
    def __init__(self, data, indexs):
        self.data = data
        self.indexs = indexs

    def __len__(self):
        return len(self.indexs)

    def __getitem__(self, index):
        feature_vec, label = self.data[self.indexs[index]]
        return feature_vec, label


class Optimizer(object):
    def __init__(self, args, data, indexs):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.data_loader = DataLoader(Dataset(data, indexs), batch_size = self.args.local_bs, shuffle = True)

    def train(self, net, intensity):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr = self.args.lr, momentum = self.args.momentum)

        sum_loss = .0
        for i in range(intensity):
            batch_loss = .0
            cnt = 0
            for batch_idx, (feature_vec, labels) in enumerate(self.data_loader):
                feature_vec, labels = feature_vec.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                log_probs = net(feature_vec)
                loss = self.loss_func(log_probs, labels)
                batch_loss += loss.item()
                loss.backward()
                optimizer.step()
                cnt += 1
            sum_loss += batch_loss / cnt
        return net.state_dict(), sum_loss / intensity
        
    def test(self, net):
        test_loss = .0
        correct = 0
        net.eval()
        for batch_idx, (feature_vec, labels) in enumerate(self.data_loader):
            feature_vec, labels = feature_vec.to(self.args.device), labels.to(self.args.device)
            log_probs = net(feature_vec)
            test_loss += F.cross_entropy(log_probs, labels, reduction = 'sum').item()
            _, y_pred = torch.max(log_probs, 1)
            correct += (y_pred == labels).sum().item()
        test_acc = correct * 100.0 / len(self.data_loader.dataset)
        test_loss /= len(self.data_loader.dataset)
        return test_acc, test_loss
        