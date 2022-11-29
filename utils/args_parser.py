import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # federated learning arguments
    parser.add_argument('--rounds', type = int, default = 50, help = "number of communication rounds")
    parser.add_argument('--num_users', type = int, default = 100, help = "number of users: N")
    parser.add_argument('--frac', type = float, default = 0.1, help = "the fraction of clients for every epoch: frac")
    parser.add_argument('--local_ep', type = int, default = 5, help = "local training epoch: bs")
    parser.add_argument('--local_bs', type = int, default = 128, help = "local batch size: bs")
    parser.add_argument('--lr', type = float, default = 0.001, help = "learning rate")
    parser.add_argument('--momentum', type = float, default = 0.5, help = "SGD momentum (default: 0.5)")
    parser.add_argument('--modeltype', type = str, default = 'cnn', help = 'machine learning model')

    # dataset arguments
    parser.add_argument('--dataset', type = str, default = 'mnist', help = 'dataset name')
    parser.add_argument('--iid', action = 'store_true', help = 'whether i.i.d or not')
    parser.add_argument('--uniform', action = 'store_true', help = 'whether uniform or not')
    
    # running arguments
    parser.add_argument('--gpu', type = int, default = -1, help = "GPU ID, -1 for CPU")
    parser.add_argument('--seed', type = int, default = 10, help = 'random seed (default: 10)')

    args = parser.parse_args()
    return args
