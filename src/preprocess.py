#!/usr/bin/env python

import os
import pickle as pkl
import json
import yaml
from collections import Counter
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from scipy import sparse

# from pytorch_transformers.tokenization_bert import BertTokenizer
from transformers import AutoTokenizer 
import logging
import random
from loguru import logger
from box import Box
from collections import defaultdict


class Preprocessor:

    def __init__(self, args, config_name=None):
        self.args = args

        basename = os.path.basename(os.getcwd())
        config = Box(yaml.load(open('src/config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader))

        self.data_dir = config.data_dir.format(basename)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
        self.config = config
        # self.special_token = ['\ufeff']

    def have_labeled(self, sentence, start, end):
        '''
        if sentence[start] == 'B' and all([sentence[i] == 'I' for i in range(start + 1, end + 1)]):
            return False
        '''
        if all([sentence[i] == 'O' for i in range(start, end + 1)] ):
            return False
        return True

    def mytokenize(self, line):
        line = line.replace('\t', '。')
        location_dict = {}
        tokens = self.tokenizer.tokenize(line)
        i, j = 0, 0
        while i < len(tokens) or j < len(line):
            if tokens[i] == line[j].lower() or tokens[i] == self.config.UNK:
                location_dict[j] = i
                i += 1
                j += 1
            else:
                tmp_length = len(tokens[i].replace('#', ''))
                #print(line[j:j+tmp_length], tokens[i])
                #print("line", ord(line[j]))
                assert line[j:j+tmp_length].lower() == tokens[i].replace('#', '')
                for k in range(tmp_length):
                    location_dict[j+k] = i
                j += tmp_length
                i += 1
        return tokens, location_dict

    def transform2indices(self, mode='train'):
        filename = os.path.join(self.config.dirname, '{}.json'.format(mode))
        with open(filename, 'r', encoding='utf-8') as f:
            data = f.read().splitlines()
        res = []
        logger.info("start preprocessing ... {} data".format(mode))
        for line in tqdm(data):
            line = json.loads(line)
            doc, doc_id = line['document'] , line['doc_id']
            doc_indicator, doc_token, words = [], [], []
            for sentence in doc:
                sent_indicator = []
                sent_token = []
                for word in sentence:
                    token = self.tokenizer.tokenize(word)
                    if len(token) == 0:
                        if word == ' ' or word in self.config.special_token:
                            token = [' ']
                        else:
                            raise Exception("Unexpected token: {}".format(word))
                    sent_indicator += [1] + [0] * (len(token) - 1)
                    sent_token += token
                doc_token += sent_token
                doc_indicator += sent_indicator
                assert len(doc_token) == len(doc_indicator)
                words += sentence
            max_length = len(words)
            assert len(doc_token) <= 510

            input_tokens = ['[CLS]'] + doc_token + ['[SEP]']
            input_id = self.tokenizer.convert_tokens_to_ids(input_tokens)
            input_mask = [1] * len(input_id)
            input_segment = [0] * len(input_id)

            doc_indicator = [0] + doc_indicator + [0]

            if mode == 'test':
                chains = []
            else:
                chains = line['chain']
                chains = [[(s, e, entity) for s, e, entity in chain if e < max_length] for chain in chains]
                chains = [sorted(list(set([(s, e, entity) for s, e, entity in chain])), key=lambda x:x[0]) for chain in chains]
            res.append((doc_id, input_id, input_mask, input_segment, chains, doc_indicator, words))
        return res
            # pkl.dump((input_ids, input_masks, input_segments, input_labels, mention_set, input_poses), path)

    def manage(self):
        print("start building source file")
        train_data = self.transform2indices('train')
        valid_data = self.transform2indices('valid')
        test_data = self.transform2indices('test')
        res = (train_data, valid_data, test_data)
        result_file = os.path.join(self.config.dirname, 'data.pkl')
        with open(result_file, 'wb') as f:
            pkl.dump(res, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', default='notest')
    parser.add_argument('--doc', default='doc')
    args = parser.parse_args()
    preprocessor = Preprocessor(args)
    preprocessor.manage()
