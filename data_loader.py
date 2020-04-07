import os
import torch
from collections import defaultdict
import numpy as np

class Tree():
    def __init__(self, parent=None, level=0):
        self.parent = parent
        self.level = level
        self.children = list()

        
class RumorDataLoader():
    def __init__(self, inp_files, device):
        self.inp_files = inp_files
        self.device = device
        self.trees, self.labels = self.load_data()

    def load_data(self):
        trees = {}
        labels = {}
        tree_input = os.path.join(self.inp_files['source_folder'], self.inp_files['data_x'])
        tree_label = os.path.join(self.inp_files['source_folder'], self.inp_files['data_y'])

        with open(tree_label, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                if len(line) <= 0:
                    continue
                label, eid = line[0], line[2]
                assert(eid not in label)
                labels[eid] = label
        
        with open(tree_input, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                if len(line) <= 0:
                    continue
                eid, indexP, indexC, degree, maxL, vec = line
                if eid not in labels:
                    continue
                if eid not in trees:
                    trees[eid] = {}
                trees[eid][indexC] = {
                    'parent': indexP, 'degree': degree, 
                    'maxL': maxL, 'vec': vec 
                }
        return trees, labels

    def generate_fold_data(self, fold_index):
        assert(fold_index in (150, 151, 152, 153, 154, 160, 161, 162, 163, 164))
        def extract_dataset(filename, filetype='test'):
            tree = {}
            label = {}
            df = {}
            D = 0
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue
                    if line not in self.trees or line not in self.labels:
                        continue
                    tree[line] = self.trees[line]
                    label[line] = self.labels[line]
                    if filetype == 'train':
                        D += len(tree[line])                       
                        for child in tree[line]:
                            vec = tree[line][child]['vec']
                            words = set([int(word_info.split(":")[0]) for word_info in vec.split()])
                            # document frequency
                            for word_idx in words:
                                if word_idx not in df:
                                    df[word_idx] = 0
                                df[word_idx] += 1

            assert(len(tree) == len(label))
            return (tree, label), df, D

        train_file = os.path.join(self.inp_files['fold_folder'], 'RNNtrainSet_Twitter%d_tree.txt'%fold_index)
        test_file = os.path.join(self.inp_files['fold_folder'], 'RNNtestSet_Twitter%d_tree.txt'%fold_index)
        train_dataset, df, D = extract_dataset(train_file, 'train')
        test_dataset, _, _ = extract_dataset(test_file)

        sorted_word_idx = sorted(list(df.keys()))
        word_idf = {word_idx: np.log(D/df[word_idx]) for word_idx in df}
        idf = np.array([word_idf[word] for word in sorted_word_idx])
        vocab = {word_idx: idx for idx, word_idx in enumerate(sorted_word_idx)}

        train_data = self.processing(train_dataset, vocab, idf)
        test_data = self.processing(test_dataset, vocab, idf)
        return train_data, test_data, len(vocab)

    def processing(self, dataset, vocab, idf):
        trees, labels = dataset
        all_adj = {}
        all_tfidf = {}
        all_level = {}
        all_labels = {}

        label_mapping = {
            'true': 0, 'false': 1, 'non-rumor': 2, 'unverified': 3       
        }
        for eid in trees:
            eid_tree = trees[eid]
            eid_label = labels[eid]
            children_idx = sorted(list(eid_tree))
            tree_info = [Tree() for _ in range(len(children_idx))]
            children_mapping = {child_idx:idx for idx, child_idx in enumerate(children_idx)}
            adj = np.zeros((len(children_idx), len(children_idx)))
            tfidf_matrix = np.zeros((len(children_idx), len(vocab)))
            root = None
            for i, child_idx in enumerate(children_idx):
                child = eid_tree[child_idx]
                # tf-idf
                tf = np.zeros((len(vocab)))
                vec = child['vec']
                freq = 0
                word_tf = {}
                for word_info in vec.split():
                    word_info = word_info.split(":")
                    word_idx, word_freq = word_info
                    word_idx = int(word_idx)
                    word_freq = int(word_freq)
                    freq += word_freq
                    word_tf[word_idx] = word_freq
                word_tf = {word_idx:word_tf[word_idx]/freq for word_idx in word_tf}
                for word_idx in word_tf:
                    if word_idx not in vocab:
                        continue
                    tf[vocab[word_idx]] = word_tf[word_idx]
                tfidf = tf * idf
                tfidf_matrix[children_mapping[child_idx]] = tfidf
                # adjancy matrix
                parent = child['parent']
                if parent != 'None':
                    adj[children_mapping[parent], children_mapping[child_idx]] = 1
                if parent == 'None':
                    root = tree_info[i]
                else:
                    tree_info[i].parent = tree_info[children_mapping[parent]]
                    tree_info[children_mapping[parent]].children.append(tree_info[i])
            queue = [root]
            while len(queue) > 0:
                node, queue = queue[0], queue[1:]
                if node.parent is not None:
                    node.level = node.parent.level + 1
                else:
                    node.level = 0
                queue.extend(node.children)
            level = [tree_i.level for tree_i in tree_info]

            all_adj[eid] = torch.from_numpy(adj).float()
            all_tfidf[eid] = torch.from_numpy(tfidf_matrix).float()
            all_labels[eid] = torch.from_numpy(np.array(label_mapping[eid_label])).long().unsqueeze(0)
            all_level[eid] = torch.from_numpy(np.array(level)).float()
        return all_adj, all_tfidf, all_level, all_labels

if __name__ == "__main__":
    inp_files = {
        'source_folder': '../data/RvNN_resource/', 
        'data_x': 'data.BU_RvNN.vol_5000.txt', 
        'data_y': 'Twitter15_label_All.txt', 
        'fold_folder': '../data/nfold/'
    }
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataloader = RumorDataLoader(inp_files, device)
    dataloader.generate_fold(150)