import torch
import argparse
import numpy as np
import random
from data_loader import RumorDataLoader
from wrapper import Wrapper
# 15 TD
# Avg Label true		P 0.841	R 0.830	F 0.835
# Avg Label false		P 0.774	R 0.739	F 0.753
# Avg Label non-rumor		P 0.732	R 0.703	F 0.715
# Avg Label unverified	P 0.716	R 0.782	F 0.746
# Avg Accuracy			0.762
# Avg Macro			P 0.766	R 0.763	F 0.762
# 16 TD
# Avg Label true		P 0.879	R 0.889	F 0.878
# Avg Label false		P 0.828	R 0.807	F 0.816
# Avg Label non-rumor		P 0.681	R 0.634	F 0.655
# Avg Label unverified	P 0.732	R 0.805	F 0.761
# Avg Accuracy			0.782
# Avg Macro			P 0.780	R 0.784	F 0.777
# =====================================================
# 15 BU
# Avg Label true		P 0.768	R 0.784	F 0.775
# Avg Label false		P 0.751	R 0.713	F 0.726
# Avg Label non-rumor		P 0.697	R 0.732	F 0.712
# Avg Label unverified	P 0.707	R 0.683	F 0.691
# Avg Accuracy			0.726
# Avg Macro			P 0.731	R 0.728	F 0.726

# 16 BU
# Avg Label true		P 0.801	R 0.816	F 0.805
# Avg Label false		P 0.748	R 0.810	F 0.776
# Avg Label non-rumor		P 0.647	R 0.563	F 0.595
# Avg Label unverified	P 0.676	R 0.666	F 0.667
# Avg Accuracy			0.724
# Avg Macro			P 0.718	R 0.714	F 0.711

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='TD_RvNN')
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--dataset', type=int, default=15)
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--n_class', type=int, default=4)
parser.add_argument('--n_epoch', type=int, default=20)
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
    seeds(args.seed)
    main()
