import scipy
import numpy as np
import torch
from scipy.spatial import distance


def cos_cdist(matrix, v):
    return scipy.spatial.distance.cdist(matrix, v, 'cosine').T


def top_k(y_pred, vec_matrix, k):
    tmp = cos_cdist(vec_matrix, y_pred)
    total_top_k = np.argsort(tmp)[:, :k]
    return total_top_k


def calc_evaluation(top_k_results, hashtag_y, idx2word, one_shot=False, precision_k=5, recall_k=5, accuracy_k=5):
    gt_hashtag, pred_hashtag = [], []
    precision, recall, accuracy = [], [], []

    if not one_shot:
        for hashtag in hashtag_y:
            hashtag = hashtag.strip().split()
            gt_hashtag.append(hashtag)

    else:
        for hashtag in hashtag_y:
            gt_hashtag.append(idx2word(hashtag))

    for row in range(len(top_k_results)):
        tmp = []
        for hashtag in top_k_results[row]:
            tmp += [idx2word[hashtag]]
        pred_hashtag.append(tmp)

    for row in range(len(pred_hashtag)):
        intersection_prec = len(set(pred_hashtag[row][:precision_k]) & set(gt_hashtag[row]))
        intersection_recall = len(set(pred_hashtag[row][:recall_k]) & set(gt_hashtag[row]))
        intersection_accuracy = len(set(pred_hashtag[row][:accuracy_k]) & set(gt_hashtag[row]))
        total_gt = len(set(gt_hashtag[row]))
        total_result = len(set(pred_hashtag[row]))

        precision.append(intersection_prec / total_result)
        recall.append(intersection_recall / total_gt)
        accuracy.append(1 if intersection_accuracy != 0 else 0)

    total_prec, total_recall, total_acc = 0, 0, 0
    for row in range(len(precision)):
        total_prec += precision[row]
        total_recall += recall[row]
        total_acc += accuracy[row]

    total_prec /= len(precision)
    total_recall /= len(recall)
    total_acc /= len(recall)
    return total_prec * 100, total_recall * 100, total_acc * 100


def calc_accuracy(top_k_results, hashtag_y, idx2word, one_shot=False):

    if not one_shot:
        gt_hashtag, pred_hashtag = [], []
        true_positive, total = 0, 0

        # top_k_results = np.concatenate((obj_top_k, scn_top_k), axis=1)

        for hashtag in hashtag_y:
            hashtag = hashtag.strip().split()
            gt_hashtag.append(hashtag)
        for row in range(len(top_k_results)):
            tmp = []
            for hashtag in top_k_results[row]:
                tmp += [idx2word[hashtag]]
            pred_hashtag.append(tmp)
        for row in range(len(pred_hashtag)):
            true_positive += len(set(pred_hashtag[row]) & set(gt_hashtag[row]))
            total += len(gt_hashtag[row])
        return true_positive, total
    else:
        tmp = np.empty(shape=(len(hashtag_y), len(hashtag_y[0])), dtype=bool)
        for i in range(len(top_k_results)):
            result = top_k_results[i]
            for positive in result:
                tmp[i][positive] = True

        epsilon = 1e-7
        gt_hashtag = torch.from_numpy(tmp)
        pred_hashtag = hashtag_y.bool()
        true_positive = (pred_hashtag & gt_hashtag).sum().float()
        false_positive = (pred_hashtag & (~gt_hashtag)).sum().float()
        recall = torch.mean(true_positive / (true_positive + false_positive + epsilon))
        return recall







