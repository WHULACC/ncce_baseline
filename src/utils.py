#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import pickle as pkl
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import numpy as np
from src.preprocess import Preprocessor

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class MyDataLoader:
    def __init__(self, config):
        self.config = config 

        result_file = os.path.join(config.dirname, 'data.pkl')
        preprocessor = Preprocessor(config)
        if not os.path.exists(result_file):
            preprocessor.manage()

        self.data = pkl.load(open(result_file, 'rb'))
    
    def process_chains(self, chains, indicator):
        bio_list, new_chains = [], []
        for i in range(len(chains)):
            bio = [0] * sum(indicator[i])
            chain = chains[i]
            for chain_id in range(len(chain)):
                cur_chain = chain[chain_id]
                for s, e, entity in cur_chain:
                    if any(bio[k] == 1 for k in range(s, e+1)): continue
                    bio[s] = self.config.bio_dict['B']
                    for k in range(s+1, e+1):
                        bio[k] = self.config.bio_dict['I']
            new_chains.append([[z[:2] for z in line] for line in chain])
            bio_list.append(bio)
        return bio_list, new_chains
    
    def collate_fn(self, lst):
        doc_ids, input_ids, input_masks, input_segments, chains, indicator, _ = zip(*lst)

        if self.mode == 'test':
            bio_list, new_chains = [], []
        else:
            bio_list, new_chains = self.process_chains(chains, indicator)

        res = {
            'doc_ids': doc_ids,
            'input_ids': input_ids,
            'input_masks': input_masks,
            'input_segments': input_segments,
            'seq_indicator': indicator,
            'chains': new_chains,
            'bio_list': bio_list
        }
        nocuda = ['chains', 'doc_ids']
        res = {k : 
                v if k in nocuda else torch.tensor(v).to(self.config.device) 
                for k, v in res.items()}
        return res
    
    def collate_fn_wrapper(self, mode):
        def wrapper(lst):
            self.mode = mode
            return self.collate_fn(lst)
        return wrapper

    def getdata(self):
        modes = ['train', 'valid', 'test']
        res = []
        for i, mode in enumerate(modes):
            data = DataLoader(MyDataset(self.data[i]), shuffle=False, batch_size=self.config.batch_size, collate_fn=self.collate_fn_wrapper(mode))
            res.append(data)
        return res 

def get_indices(bios):
    res = []
    start, end = -1, -1
    for i, word in enumerate(bios):
        if word == 'B':
            if start != -1:
                res.append((start, end))
            start, end = i, i
        elif word == 'O':
            if start != -1:
                res.append((start, end))
            start, end = -1, -1
        else:
            end = i
    if start != -1:
        res.append((start, end))
    return res

class MentionMetric:
    def __init__(self, id2tag) -> None:
        self.tp, self.fp, self.fn = 0, 0, 0
        self.word_dict = id2tag

    def add_instance(self, predict_out, gold_bio):
        predict_out = predict_out.argmax(-1)

        predict_bio = [self.word_dict[w.item()] for w in predict_out]
        gold_bio = [self.word_dict[w.item()] for w in gold_bio]

        predict_bio = get_indices(predict_bio)
        gold_bio = get_indices(gold_bio)
        self.fn += len(gold_bio)
        self.fp += len(predict_bio)
        for w in predict_bio:
            for z in gold_bio:
                if w[0] == z[0] and w[1] == z[1]:
                    self.tp += 1
    def compute_f1(self):
        p = self.tp / self.fp if self.fp > 0 else 0
        r = self.tp / self.fn if self.fn > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        return (p, r, f1), (self.tp, self.fp, self.fn)
    def clear(self):
        self.tp, self.fp, self.fn = 0, 0, 0

import os
import numpy as np

def getPredictCluster(predict_indices, mention_interaction):
    """
    mention_interaction = [
        [[1.3,0.5], [0.1, 0.3], [1, 0.2],[0,0],[1,2]],
        [[0.3,0.5], [1.1, 0.3], [1, 0.2],[0,0],[1,2]],
        [[1.3,0.5], [0.1, 0.3], [1, 0.2],[0,0],[1,2]],
        [[0.3,0.5], [1, 0.3], [1.1, 0.2],[1,0],[1,2]],
        [[0.3,0.5], [1.1, 0.3], [0, 0.2],[0,1],[3,2]],
    ]
    predict_indices = [(1,2), (3, 4), (7, 8)]
    :param predict_indices:
    :param mention_interaction:
    :return:
    
    clusters:
    mention_to_predict:
    """
    mention_interaction = mention_interaction.cpu().tolist()
    exp = np.exp(mention_interaction)
    mention_interaction = exp / np.sum(exp, -1, keepdims=True)
    
    cluster = dict()
    
    cluster_id, idx = [0 for _ in mention_interaction], 1
    
    for i in range(len(mention_interaction)):
        indices = predict_indices[i]
        if i == 0:
            continue
        label = np.argmax(mention_interaction[i, :i], -1)
        if sum(label) == 0:
            cluster_id[i] = idx
            idx += 1
            continue
        ancestor = mention_interaction[i, :i, 1].argmax()
        cluster_id[i] = cluster_id[ancestor]
    
    cluster_id_dict = {}
    for i, index in enumerate(cluster_id):
        if index in cluster_id_dict:
            cluster_id_dict[index].append(i)
        else:
            cluster_id_dict[index] = [i]
    
    clusters = []
    mention_to_predict = {}
    for k, v in cluster_id_dict.items():
        cluster = [predict_indices[w] for w in v]
        cluster = tuple(tuple(w) for w in cluster)
        clusters.append(cluster)
        for w in cluster:
            mention_to_predict[w] = cluster
    
    return clusters, mention_to_predict


def evaluate_coref(predict_indices, mention_interaction, gold_mention_set, evaluator):
    
    gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_mention_set]
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            mention_to_gold[mention] = gc
    predicted_clusters, mention_to_predicted = getPredictCluster(predict_indices, mention_interaction)
    
    evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
    return predicted_clusters, gold_clusters


if __name__ == '__main__':
    mention_interaction = [
        [[0.3,0.5], [0.1, 0.3], [1, 0.2],[0,0],[1,2]],
        [[1.5,0.5], [0.1, 0.3], [1, 0.2],[0,0],[1,2]],
        [[0.3,0.5], [1.1, 0.3], [0, 0.2],[0,0],[1,2]],
        [[1.3,0.5], [0.1, 1.3], [0.1, 2.2],[-1,0],[1,2]],
        [[1.3,0.5], [1.1, 1.3], [1, 0.2],[2,1],[0,2]],
    ]
    predict_indices = [(1,2), (3, 4), (7, 8), (10,11), (12,13)]
