from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import Counter
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)

class CorefEvaluator(object):
    def __init__(self):
        self.evaluators = [Evaluator(m) for m in (muc, b_cubed, ceafe)]
        self.nums_count = {i:{'tp':0, 'fp':0, 'fn':0} for i in range(100)}

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        for e in self.evaluators:
            e.update(predicted, gold, mention_to_predicted, mention_to_gold)
        predicted = [w for line in predicted for w in line]
        gold = [w for line in gold for w in line]

        for x in predicted:
            l = x[1] - x[0]
            if l in self.nums_count:
                self.nums_count[l]['fp'] += 1
            else:
                self.nums_count[l] = {}
                self.nums_count[l]['fp'] = 1
                self.nums_count[l]['fn'] = 0
                self.nums_count[l]['tp'] = 0

        for x in gold:
            l = x[1] - x[0]
            if l in self.nums_count:
                self.nums_count[l]['fn'] += 1
            else:
                self.nums_count[l] = {}
                self.nums_count[l]['fn'] = 1
                self.nums_count[l]['fp'] = 0
                self.nums_count[l]['tp'] = 0

        for x in predicted:
            for y in gold:
                if x == y:
                    l = x[1] - x[0]
                    self.nums_count[l]['tp'] += 1

    def get_f1(self):
        return sum(e.get_f1() for e in self.evaluators) / len(self.evaluators)

    def get_recall(self):
        return sum(e.get_recall() for e in self.evaluators) / len(self.evaluators)

    def get_precision(self):
        return sum(e.get_precision() for e in self.evaluators) / len(self.evaluators)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()
    
    def get_all_score(self):
        precision = [e.get_precision() for e in self.evaluators]
        recalls = [e.get_recall() for e in self.evaluators]
        f1_score = [e.get_f1() for e in self.evaluators]
        res = list(zip(precision, recalls, f1_score))
        avg_score = self.get_prf()
        res = {
            'muc': res[0],
            'b_cubed': res[1],
            'ceafe': res[2],
            'avg': avg_score
        }
        return res

    def get_self_prf(self, count):
        tp, fp, fn = count['tp'], count['fp'], count['fn']
        p = tp / fp if fp > 0 else 0
        r = tp / fn if fn > 0 else 0
        f = 2 * p * r / (p + r) if p + r > 0 else 0
        return fn, p, r, f


    def get_res(self):
        keys = sorted(list(self.nums_count))

        res = []
        for key in keys:
            tmp = self.get_self_prf(self.nums_count[key])
            res.append((key, *tmp))
        import pickle as pkl
        #pkl.dump(res, open('count_res.pkl', 'wb'))
        return res



class Evaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(predicted, gold)
        else:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den


def evaluate_documents(documents, metric, beta=1):
    evaluator = Evaluator(metric, beta=beta)
    for document in documents:
        evaluator.update(document)
    return evaluator.get_precision(), evaluator.get_recall(), evaluator.get_f1()


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            if len(c2) != 1:
                correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    row_ind, col_ind = linear_sum_assignment(-scores)
    similarity = sum(scores[row_ind, col_ind])
    return similarity, len(clusters), similarity, len(gold_clusters)

# def ceafe(clusters, gold_clusters):
#     clusters = [c for c in clusters if len(c) != 1]
#     scores = np.zeros((len(gold_clusters), len(clusters)))
#     for i in range(len(gold_clusters)):
#         for j in range(len(clusters)):
#             scores[i, j] = phi4(gold_clusters[i], clusters[j])
#     matching = linear_assignment(-scores)
#     similarity = sum(scores[matching[:, 0], matching[:, 1]])
#     return similarity, len(clusters), similarity, len(gold_clusters)


def lea(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1:]:
                    if m2 in mention_to_gold and mention_to_gold[m] == mention_to_gold[m2]:
                        common_links += 1

        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem
