#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
import json

from src.utils import MyDataLoader
from src.model import myLSTM
from src.metrics import CorefEvaluator
from src.utils import MentionMetric, evaluate_coref, getPredictCluster
from box import Box

import random
import numpy as np

random.seed(42)
torch.random.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)

class Pipeline:
    def __init__(self, args):
        self.args = args
        basename = 'scripts'

        config = Box(yaml.load(open('src/config.yaml', 'r', encoding='utf-8').read(), Loader=yaml.FullLoader))

        config.device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
        config.id2tag = {k : v for k, v in enumerate(config.bio_list)}
        config.bio_dict = {v: k for k, v in config.id2tag.items()}

        config.data_dir = config.data_dir.format(basename)
        config.target_dir = config.target_dir.format(basename)
        if not os.path.exists(config.target_dir):
            os.makedirs(config.target_dir)
        self.config = config
    
    def save_to_file(self):
        pass

    def test(self):
        self.model.eval()
        dataloader = self.testLoader

        dataiter = tqdm(dataloader, total=dataloader.__len__(), file=sys.stdout)

        res = []

        for index, data in enumerate(dataiter):
            with torch.no_grad():
                bio_p, predict_indices, mention_interaction = self.model.inference(**data)
                predicted_clusters, mention_to_predicted = getPredictCluster(predict_indices, mention_interaction)
                res.append({'doc_id': data['doc_ids'][0], 'chain': [list(w) for w in predicted_clusters]})
        
        output_path = './data/submit/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        output_file = os.path.join(output_path, 'submit.jsonl')

        with open(output_file, 'w', encoding='utf-8') as f:
            for line in res:
                json.dump(line, f, ensure_ascii=False)
                f.write('\n')


    def main(self):
        self.trainLoader, self.validLoader, self.testLoader = MyDataLoader(self.config).getdata()
        cfg = self.config
        self.mention_metric = MentionMetric(self.config.id2tag)
        
        self.model = myLSTM(self.config).to(self.config.device)
        self.model.load_state_dict(torch.load(os.path.join(cfg.target_dir, self.args.chk))['model'])
        self.model.eval()

        self.criterion = nn.CrossEntropyLoss()
        self.test()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=0)
    parser.add_argument('--chk', '--checkpoint', default='best_12.pth.tar', type=str, help="checkpoint path: ./data/save/xxx")
    args = parser.parse_args()
    pipeline = Pipeline(args)
    pipeline.main()
