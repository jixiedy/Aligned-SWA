#coding=utf-8
from __future__ import print_function, absolute_import
import numpy as np
import copy
from collections import defaultdict
import sys

import torch
from sklearn.metrics import average_precision_score

def eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, N=100):
    """Evaluation with cuhk03 metric
    Key: one image for each gallery identity is randomly sampled for each query identity.
    Random sampling is performed N times (default: N=100).
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    # 按行排序并在输出的张量上写入原来元素的索引
    indices = np.argsort(distmat, axis=1)
    # [:, np.newaxis]: 在列上增加维度；　[np.newaxis,:]：在行上增加维度　
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        # np.invert: 计算位非
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        kept_g_pids = g_pids[order][keep]
        g_pids_dict = defaultdict(list)
        for idx, pid in enumerate(kept_g_pids):
            g_pids_dict[pid].append(idx)

        cmc, AP = 0., 0.
        for repeat_idx in range(N):
            mask = np.zeros(len(orig_cmc), dtype=np.bool)
            for _, idxs in g_pids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                mask[rnd_idx] = True
            masked_orig_cmc = orig_cmc[mask]
            # .cumsum()返回给定axis上的累计和, 函数的原型如下：
            """
            numpy.cumsum(a, axis=None, dtype=None, out=None)
            Return the cumulative sum of the elements along a given axis.
            """
            _cmc = masked_orig_cmc.cumsum()
            _cmc[_cmc > 1] = 1
            cmc += _cmc[:max_rank].astype(np.float16)
            # compute AP
            num_rel = masked_orig_cmc.sum()
            tmp_cmc = masked_orig_cmc.cumsum()
            tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * masked_orig_cmc
            AP += tmp_cmc.sum() / num_rel
        cmc /= N
        AP /= N
        all_cmc.append(cmc)
        all_AP.append(AP)
        num_valid_q += 1.

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float16)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

# def eval_noncamid(distmat, q_pids, g_pids, q_camids=None, g_camids=None, max_rank=50):
#     """Evaluation with noncamid_dataset metric
#     Key: for each query identity, its gallery images from the same camera view are discarded.
#     """
#     num_q, num_g = distmat.shape
#     if num_g < max_rank:
#         max_rank = num_g
#         print("Note: number of gallery samples is quite small, got {}".format(num_g))
#     indices = np.argsort(distmat, axis=1)
#     matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

#     if q_camids is None:
#         q_camids = np.zeros(num_q).astype(np.int32)
#     if g_camids is None:
#         g_camids = np.ones(num_g).astype(np.int32)
#     # q_camids = np.asarray(q_camids)
#     # g_camids = np.asarray(g_camids)

#     # compute cmc curve for each query
#     all_cmc = []
#     all_AP = []
#     num_valid_q = 0. # number of valid query
#     for q_idx in range(num_q):
#         # get query pid and camid
#         q_pid = q_pids[q_idx]
#         q_camid = q_camids[q_idx]

#         # remove gallery samples that have the same pid and camid with query
#         order = indices[q_idx]
#         remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
#         keep = np.invert(remove)

#         # compute cmc curve
#         orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
#         if not np.any(orig_cmc):
#             # this condition is true when query identity does not appear in gallery
#             continue

#         cmc = orig_cmc.cumsum()
#         cmc[cmc > 1] = 1

#         all_cmc.append(cmc[:max_rank])
#         num_valid_q += 1.

#         # compute average precision
#         # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
#         num_rel = orig_cmc.sum()
#         tmp_cmc = orig_cmc.cumsum()
#         tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
#         tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
#         AP = tmp_cmc.sum() / num_rel
#         all_AP.append(AP)

#     assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

#     all_cmc = np.asarray(all_cmc).astype(np.float16)
#     all_cmc = all_cmc.sum(0) / num_valid_q
#     # print('all_cmc', all_cmc)
#     mAP = np.mean(all_AP)

#     return all_cmc, mAP

# def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, use_metric_cuhk03=False, use_metric_noncamid=False):
def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=100, use_metric_cuhk03=False):
    if use_metric_cuhk03:
        return eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
    # elif use_metric_noncamid:
    #     return eval_noncamid(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
    else:
        return eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)

# def _unique_sample(ids_dict, num):
#     mask = np.zeros(num, dtype=np.bool)
#     for _, indices in ids_dict.items():
#         i = np.random.choice(indices)
#         mask[i] = True
#     return mask


# def Cmc(distmat, query_ids=None, gallery_ids=None,
#         query_cams=None, gallery_cams=None, topk=50,
#         separate_camera_set=False,
#         single_gallery_shot=False,
#         first_match_break=False):
#     m, n = distmat.shape
#     # Fill up default values
#     if query_ids is None:
#         query_ids = np.arange(m)
#     if gallery_ids is None:
#         gallery_ids = np.arange(n)
#     if query_cams is None:
#         query_cams = np.zeros(m).astype(np.int32)
#     if gallery_cams is None:
#         gallery_cams = np.ones(n).astype(np.int32)
#     # Ensure numpy array
#     query_ids = np.asarray(query_ids)
#     gallery_ids = np.asarray(gallery_ids)
#     query_cams = np.asarray(query_cams)
#     gallery_cams = np.asarray(gallery_cams)
#     # Sort and find correct matches
#     indices = np.argsort(distmat, axis=1)
#     matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
#     # Compute CMC for each query
#     ret = np.zeros(topk)
#     num_valid_queries = 0
#     for i in range(m):
#         # Filter out the same id and same camera
#         valid = ((gallery_ids[indices[i]] != query_ids[i]) |
#                  (gallery_cams[indices[i]] != query_cams[i]))
#         if separate_camera_set:
#             # Filter out samples from same camera
#             valid &= (gallery_cams[indices[i]] != query_cams[i])
#         if not np.any(matches[i, valid]):
#             continue
#         if single_gallery_shot:
#             repeat = 10
#             gids = gallery_ids[indices[i][valid]]
#             inds = np.where(valid)[0]
#             ids_dict = defaultdict(list)
#             for j, x in zip(inds, gids):
#                 ids_dict[x].append(j)
#         else:
#             repeat = 1
#         for _ in range(repeat):
#             if single_gallery_shot:
#                 # Randomly choose one instance for each id
#                 sampled = (valid & _unique_sample(ids_dict, len(valid)))
#                 index = np.nonzero(matches[i, sampled])[0]
#             else:
#                 index = np.nonzero(matches[i, valid])[0]
#             delta = 1. / (len(index) * repeat)
#             for j, k in enumerate(index):
#                 if k - j >= topk:
#                     break
#                 if first_match_break:
#                     ret[k - j] += 1
#                     break
#                 ret[k - j] += delta
#         num_valid_queries += 1
#     if num_valid_queries == 0:
#         raise RuntimeError("No valid query")
#     return ret.cumsum() / num_valid_queries


# def mean_ap(distmat, query_ids=None, gallery_ids=None,
#             query_cams=None, gallery_cams=None):
#     m, n = distmat.shape
#     # Fill up default values
#     if query_ids is None:
#         query_ids = np.arange(m)
#     if gallery_ids is None:
#         gallery_ids = np.arange(n)
#     if query_cams is None:
#         query_cams = np.zeros(m).astype(np.int32)
#     if gallery_cams is None:
#         gallery_cams = np.ones(n).astype(np.int32)
#     # Ensure numpy array
#     query_ids = np.asarray(query_ids)
#     gallery_ids = np.asarray(gallery_ids)
#     query_cams = np.asarray(query_cams)
#     gallery_cams = np.asarray(gallery_cams)
#     # Sort and find correct matches
#     indices = np.argsort(distmat, axis=1)
#     matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
#     # Compute AP for each query
#     aps = []
#     for i in range(m):
#         # Filter out the same id and same camera
#         valid = ((gallery_ids[indices[i]] != query_ids[i]) |
#                  (gallery_cams[indices[i]] != query_cams[i]))
#         y_true = matches[i, valid]
#         y_score = -distmat[i][indices[i]][valid]
#         if not np.any(y_true):
#             continue
#         aps.append(average_precision_score(y_true, y_score))
#     if len(aps) == 0:
#         raise RuntimeError("No valid query")
#     return np.mean(aps)