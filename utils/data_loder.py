import os
import torch
from collections import defaultdict

class Tree():
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()
    
    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)
    
class RumorDataLoader():
    def __init__(self, inp_files, device):
        self.inp_files = inp_files
        self.device = device
        self.tweets, self.labels, self.trees = self.load_data(inp_files['source_folder'])

    def load_data(self, source_folder):
        tweets, labels, trees = {}, {}, {}
        source_tweets = os.path.join(source_folder, 'source_tweets.txt')
        with open(source_tweets, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split()
                if len(line) == 0:
                    continue
                tweets[line[0]] = line[1:]

        source_label = os.path.join(source_folder, 'label.txt')
        with open(source_label, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split(':')
                if len(line) == 0:
                    continue
                labels[line[1]] = line[0]
        
        clean = lambda x: x.replace('[', '').replace(']', '').replace(' ', '').replace('\'', '').split(',')
        tree_folder = os.path.join(source_folder, 'tree')
        source_trees = os.listdir(tree_folder)
        for st in source_trees:
            sources = []
            targets = []
            with open(os.path.join(tree_folder, st), 'r', encoding='utf-8') as f:
                f.readline()
                for line in f.readlines():
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    source, target = line.split('->')
                    sources.append(source)
                    targets.append(target)
            source_set = set(sources)
            target_set = set(targets)
            all_tweets = source_set | target_set
            
            tweet_map = {tweet:idx for idx, tweet in enumerate(all_tweets)}
            root = None
            nodes = [Tree()] * len(all_tweets)
            heads = [0]+[tweet_map[s] for s in sources]
            for i in range(len(nodes)):
                h = heads[i]
                if h == 0:
                    root = nodes[i]
                else:
                    nodes[h-1].add_child(nodes[i])
            trees[st.replace('.txt', '')] = root
        return tweets, labels, trees
    

if __name__ == "__main__":
    inp_files = {
        'source_folder': '../data/twitter15/', 

    }
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    RumorDataLoader(inp_files, device)