import torch
import argparse
import numpy as np
import random
from data_loader import RumorDataLoader
from wrapper import Wrapper

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='TD_RvNN')
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--dataset', type=int, default=15)
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--n_class', type=int, default=4)
parser.add_argument('--n_epoch', type=int, default=500)
parser.add_argument('--lr', type=int, default=0.005)

def seeds(seed):
    # to ensure reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():

    label_mapping = {0:'true\t', 1: 'false\t', 2: 'non-rumor\t', 3: 'unverified'}
    args.label_mapping = label_mapping
    total_fold = 5

    all_result = []
    for fold in range(0, total_fold):
        model_wrapper = Wrapper(args, (loader, args.dataset*10+fold))
        fold_result = model_wrapper.run()
        print('------------Fold %d--------------'%fold)
        for key in range(len(label_mapping)):
            print('Label %s\tP %0.3f\tR %0.3f\tF %0.3f' % (label_mapping[key], *fold_result[key]))
        print('Accuracy\t\t\t%0.3f' % fold_result[-2])
        print('Macro\t\t\tP %0.3f\tR %0.3f\tF %0.3f' % fold_result[-1])
        all_result.append(fold_result)
    print('---------------------------------')
    for key in range(len(label_mapping)):
        avg_p = sum([all_result[fold][key][0] for fold in range(total_fold)]) / total_fold
        avg_r = sum([all_result[fold][key][1] for fold in range(total_fold)]) / total_fold
        avg_f = sum([all_result[fold][key][2] for fold in range(total_fold)]) / total_fold
        print('Avg Label %s\tP %0.3f\tR %0.3f\tF %0.3f' % (label_mapping[key], avg_p, avg_r, avg_f))
    avg_acc = sum([all_result[fold][-2] for fold in range(total_fold)]) / total_fold
    avg_mp = sum([all_result[fold][-1][0] for fold in range(total_fold)]) / total_fold
    avg_mr = sum([all_result[fold][-1][1] for fold in range(total_fold)]) / total_fold
    avg_mf = sum([all_result[fold][-1][2] for fold in range(total_fold)]) / total_fold
    print('Avg Accuracy\t\t\t%0.3f' % avg_acc)
    print('Avg Macro\t\t\tP %0.3f\tR %0.3f\tF %0.3f' % (avg_mp, avg_mr, avg_mf))

if __name__ == '__main__':
    avg_fn = lambda x, index: sum([z[index] for z in x]) / len(x)

    pair_results = []
    emotion_results = []
    event_results = []

    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    inp_files = {
        'source_folder': 'data/RvNN_resource/', 
        'data_x': 'data.BU_RvNN.vol_5000.txt', 
        'data_y': 'Twitter%d_label_All.txt'%args.dataset, 
        'fold_folder': 'data/nfold/'
    }
    loader = RumorDataLoader(inp_files, args.device)
    main()
