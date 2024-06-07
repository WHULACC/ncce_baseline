#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import yaml
from tqdm import tqdm
import torch
from transformers import AdamW, get_linear_schedule_with_warmup 
import torch.nn as nn

from src.utils import MyDataLoader
from src.model import myLSTM
from src.metrics import CorefEvaluator
from src.utils import MentionMetric, evaluate_coref
from box import Box
from loguru import logger

import random
import numpy as np

random.seed(42)
torch.random.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)

class Pipeline:
    def __init__(self, args):
        self.args = args

        config = Box(yaml.load(open('src/config.yaml', 'r', encoding='utf-8').read(), Loader=yaml.FullLoader))

        config.device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
        config.id2tag = {k : v for k, v in enumerate(config.bio_list)}
        config.bio_dict = {v: k for k, v in config.id2tag.items()}

        if not os.path.exists(config.target_dir):
            os.makedirs(config.target_dir)
        self.config = config

    def execute_iter(self, training=True):
        dataloader = self.validLoader
        if training:
            self.model.train()
            dataloader = self.trainLoader
        else:
            self.model.eval()

        dataiter = tqdm(dataloader, total=dataloader.__len__(), file=sys.stdout)
        f1s, losses  = [], []
        coref_evaluator = CorefEvaluator()

        for index, data in enumerate(dataiter):
            chains = data['chains'][0]
            self.model.valid_index = index
            if training:
                loss, bio_p, bio_g, predict_indices, mention_interaction = self.model(**data)
                loss.backward()
                nn.utils.clip_grad_norm(self.model.parameters(), self.config.max_grad_norm)
                self.scheduler.step()
                self.optimizer.step()
                self.model.zero_grad()
            else:
                with torch.no_grad():
                    loss, bio_p, bio_g, predict_indices, mention_interaction = self.model(**data)

            self.mention_metric.add_instance(bio_p, bio_g)
            (p, r, f1), _ = self.mention_metric.compute_f1()

            evaluate_coref(predict_indices, mention_interaction, chains, coref_evaluator)
            chain_score = coref_evaluator.get_all_score()

            f1s.append(f1)
            losses.append(loss.item())

            description = "Epoch {}, loss:{:.3f}, label bio f1:{:.3f}, mean: p {:.4f}, r {:.4f}, f {:.4f}".format(
                self.global_epcoh, np.mean(losses), f1 * 100, *[w*100 for w in chain_score['avg']])

            dataiter.set_description(description)

        self.mention_metric.clear()
        res = {'bio': (p, r, f1)}
        res.update(chain_score)
        return res 
    
    def forward(self):
        best_score, best_iter = 0, 0
        for epoch in range(self.config.epoch_size):
            self.global_epcoh = epoch

            logger.info("Start training epoch {}".format(epoch))
            self.execute_iter()
            logger.info("Start validating epoch {}".format(epoch))
            chain_score = self.execute_iter(training=False)

            avg_f1 = chain_score['avg'][-1]
            print(f'{"".ljust(10)}\tP\tR\tF1')
            for k, v in chain_score.items():
                print('{}: {:.2f}\t{:.2f}\t{:.2f}'.format(k.ljust(10), *[w*100 for w in v]))

            if avg_f1 > best_score:
                best_score, best_iter = avg_f1, epoch
                torch.save({'epoch': epoch, 'model': self.model.cpu().state_dict(), 'best_score': best_score},
                           os.path.join(self.config.target_dir, "best_{}.pth.tar".format(epoch)))
                self.model.to(self.config.device)
                print("best score: ")
                print("score: {:.4f}".format(avg_f1))

            elif epoch - best_iter > self.config.patience:
                print("Not upgrade for {} steps, early stopping...".format(self.config.patience))
                break

    def main(self):
        self.trainLoader, self.validLoader, self.testLoader = MyDataLoader(self.config).getdata()
        cfg = self.config
        self.mention_metric = MentionMetric(self.config.id2tag)
        
        self.model = myLSTM(self.config).to(self.config.device)

        self.optimizer = AdamW(self.model.parameters(),
                       lr=float(cfg.learning_rate),
                        eps=float(cfg.adam_epsilon), weight_decay=1e-6)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps = cfg.epoch_size * self.trainLoader.__len__())

        self.criterion = nn.CrossEntropyLoss()
        self.forward()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=1)
    args = parser.parse_args()
    pipeline = Pipeline(args)
    pipeline.main()
