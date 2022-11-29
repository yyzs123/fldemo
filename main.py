import numpy as np
import random
import torch
from torch import nn
from torchvision import datasets, transforms
from utils.args_parser import parse_args
from utils.data_splitter import split_data
from utils.optimizer import Optimizer
from model.cnn import CNN
import sys, copy, datetime


if __name__ == '__main__':
    # starting time
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
    # parse args
    args = parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print("Runs on {:s}, dataset {}, iid {:d}, uniform {:d}, modeltype {}, {:s}".format(str(args.device), args.dataset, args.iid, args.uniform, args.modeltype, time_str))

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed) # set the seed for numpy
    torch.manual_seed(args.seed) # set the seed for generating random numbers
    torch.cuda.manual_seed(args.seed) # Set the seed for generating random numbers for the current GPU. 
    torch.cuda.manual_seed_all(args.seed) # set the seed for generating random numbers on all GPUs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load data
    if args.dataset == 'fmnist':
        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = [0.1307], std = [0.3081])])
        train_data = datasets.FashionMNIST(root = "./data/fmnist",
                                transform = transform,
                                train = True,
                                download = True)
        test_data = datasets.FashionMNIST(root = "./data/fmnist",
                                transform = transform,
                                train = False,
                                download = True)
    elif args.dataset == 'cifar-10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_data = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=transform)
    else:
        sys.exit(0)

    # split data
    user_data, user_data_size = split_data(train_data, args.num_users, args.dataset, args.iid, args.uniform)
    
    # build model CNN
    if args.modeltype == 'cnn':
        global_net = CNN(args.dataset)
    else:
        global_net = CNN(args.dataset)

    global_w = global_net.state_dict()

    # build optimizer used for testing
    tester = Optimizer(args = args, data = test_data, indexs = np.arange(len(test_data)))

    # some results
    train_loss = []
    train_time = []
    test_acc = []
    test_loss = []

    # some args
    k = max(int(args.frac * args.num_users), 1) # select k clients randomly every epoch

    # train global model
    for i in range(args.rounds):
        local_w, sum_loss, max_time = [], .0, .0
        np.random.seed(args.seed * (i + 1))

        # radomly select clients
        selected_users = np.random.choice(np.arange(args.num_users), k, replace = False)

        # clients' capabilities
        comp = np.array(np.random.uniform(10, 31, k))
        comm = np.array(np.random.uniform(40, 80, k))

        # intensity for clients
        intensity = np.ones(k, dtype = int) * args.local_ep
  
        # local train
        for j in range(len(selected_users)):
            user_idx = selected_users[j]
            if intensity[j] > 0:
                optimizer = Optimizer(args = args, data = train_data, indexs = user_data[user_idx])
                local_net = copy.deepcopy(global_net)
                w, loss = optimizer.train(net = local_net.to(args.device), intensity = intensity[j])
                local_w.append(copy.deepcopy(w))
                sum_loss += loss
                max_time = comp[j] * intensity[j] + comm[j]

        # update global model
        global_w = local_w[0]
        for key in global_w.keys():
            for j in range(1, len(local_w)):
                global_w[key] += local_w[j][key]
            global_w[key] = torch.div(global_w[key], len(local_w))

        global_net.train()
        global_net.load_state_dict(global_w)

        avg_loss = sum_loss / len(selected_users)

        train_loss.append(avg_loss)
        train_time.append(max_time)

        acc, loss = tester.test(global_net)
        test_acc.append(acc)
        test_loss.append(loss)

        print('Round {:d}, train_loss {:.6f}, time_cost {:.6f}, test_acc {:.6f}'.format(i, avg_loss, sum(train_time), acc))

   
    # record results
    log_name = "./results/{}_{}_iid{}_uniform{}_{}_lr{}_R{}_E{}_S{}.txt".format(time_str, args.dataset, args.iid, args.uniform, args.modeltype, args.lr, args.rounds, args.local_ep, args.seed)
    with open(log_name, 'w') as log:
        log.write('train_loss\n{}\n'.format(len(train_loss)))
        log.write(str(train_loss).replace('[', '').replace(']', '').replace(',', '') + '\n')
        log.write('train_time\n{}\n'.format(len(train_time)))
        log.write(str(train_time).replace('[', '').replace(']', '').replace(',', '') + '\n')
        log.write('test_acc\n{}\n'.format(len(test_acc)))
        log.write(str(test_acc).replace('[', '').replace(']', '').replace(',', '') + '\n')
        log.write('test_loss\n{}\n'.format(len(test_loss)))
        log.write(str(test_loss).replace('[', '').replace(']', '').replace(',', '') + '\n')

