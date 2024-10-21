import torch
import numpy as np
from torch import FloatTensor
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional, Tuple, Union
from itertools import chain

Layout = Tuple[np.ndarray, np.ndarray]

def ltwh_to_ltrb_split(bbox):
    l, t, w, h = bbox
    r = l + w
    b = t + h
    return [l, t, r, b]


def compute_iou(
    box_1: Union[np.ndarray, FloatTensor],
    box_2: Union[np.ndarray, FloatTensor],
    generalized: bool = False,
) -> Union[np.ndarray, FloatTensor]:
    # box_1: [N, 4]  box_2: [N, 4]

    if isinstance(box_1, np.ndarray):
        lib = np
    elif isinstance(box_1, FloatTensor):
        lib = torch
    else:
        raise NotImplementedError(type(box_1))

    l1, t1, r1, b1 = ltwh_to_ltrb_split(box_1.T)
    l2, t2, r2, b2 = ltwh_to_ltrb_split(box_2.T)
    a1, a2 = (r1 - l1) * (b1 - t1), (r2 - l2) * (b2 - t2)

    # intersection
    l_max = lib.maximum(l1, l2)
    r_min = lib.minimum(r1, r2)
    t_max = lib.maximum(t1, t2)
    b_min = lib.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = lib.where(cond, (r_min - l_max) * (b_min - t_max), lib.zeros_like(a1[0]))

    au = a1 + a2 - ai
    iou = ai / au

    if not generalized:
        return iou

    # outer region
    l_min = lib.minimum(l1, l2)
    r_max = lib.maximum(r1, r2)
    t_min = lib.minimum(t1, t2)
    b_max = lib.maximum(b1, b2)
    ac = (r_max - l_min) * (b_max - t_min)

    giou = iou - (ac - au) / ac

    return giou


def pair_maximum_iou(layout_true: Layout, layout_predict: Layout) -> float:
    score = 0.0
    (bi, li), (bj, lj) = layout_true, layout_predict
    N = len(bi)
    for l in list(set(li)):
        _bi = bi[np.where(li == l)]
        _bj = bj[np.where(lj == l)]

        n = len(_bi)
        m = len(_bj)

        if m == 0:
            continue
        else:
            ii, jj = np.meshgrid(range(n), range(m))
            ii, jj = ii.flatten(), jj.flatten()
            iou = compute_iou(_bi[ii], _bj[jj]).reshape(n, m)
            # note: maximize is supported only when scipy >= 1.4
            ii, jj = linear_sum_assignment(iou, maximize=True)
            score += iou[ii, jj].sum().item()

    return score / N


def __compute_maximum_iou(layouts_1_and_2: Tuple[List[Layout]]) -> np.ndarray:
    layouts_1, layouts_2 = layouts_1_and_2
    N, M = len(layouts_1), len(layouts_2)
    ii, jj = np.meshgrid(range(N), range(M))
    ii, jj = ii.flatten(), jj.flatten()
    scores = np.asarray(
        [
            pair_maximum_iou(layouts_1[i], layouts_2[j])
            for i, j in zip(ii, jj)
        ]
    ).reshape(N, M)

    ii, jj = linear_sum_assignment(scores, maximize=True)
    return scores[ii, jj]



def __get_cond2layouts(layout_list: List[Layout]) -> Dict[str, List[Layout]]:
    out = dict()
    for bs, ls in layout_list:
        cond_key = str(sorted(ls))
        if cond_key not in out.keys():
            out[cond_key] = [(bs, ls)]
        else:
            out[cond_key].append((bs, ls))

    return out

def label_maximum_iou(
    layouts_1: List[Layout],
    layouts_2: List[Layout],
):
    """
    Computes Maximum IoU [Kikuchi+, ACMMM'21]
    """
    c2bl_1 = __get_cond2layouts(layouts_1)
    keys_1 = set(c2bl_1.keys())
    c2bl_2 = __get_cond2layouts(layouts_2)
    keys_2 = set(c2bl_2.keys())
    keys = list(keys_1.intersection(keys_2))
    args = [(c2bl_1[key], c2bl_2[key]) for key in keys]
    scores = [__compute_maximum_iou(a) for a in args]

    scores = np.asarray(list(chain.from_iterable(scores)))
    if len(scores) == 0:
        return 0.0
    else:
        return scores.mean().item()


if __name__ == "__main__":

    box_1 = torch.rand([4, 4])
    box_2 = torch.rand([4, 4])
    box_2[:, :2] = box_1[:, :2]
    box_2[:, 2:] = box_1[:, 2:] + 0.15 * torch.rand_like(box_1[:, 2:])

    label_1 = torch.tensor([1, 2, 3, 1])
    layout_1 = [box_1, label_1]
    label_2 = torch.tensor([1, 2, 3, 2])
    layout_2 = [box_2, label_2]

    s = pair_maximum_iou(layout_1, layout_2)
    print(s)
    # maxiou = label_maximum_iou([layout_1, layout_2], [layout_2, layout_1])
    # print(maxiou)



    # test_layout_cond(batch_size=256, cond='c', dataset_name='rico25')


