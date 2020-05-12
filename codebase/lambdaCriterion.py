


import numpy as np
import torch


class DeltaNDCG:
    def __init__(self, module):
        if module == "numpy":
            self.compute_loss = self.compute_loss_numpy
        elif module == 'pytorch':
            self.compute_loss = self.compute_loss_torch

    def compute_loss_torch(self, scores, relevances,  iDCG, sigma, device):
        precompute_S_arr = s_tensor(relevances, sigma).to(device)
        relevances_tensor = self.make_relevance_numinator_torch(relevances)
        return  torch.mul(self.ranknet_cost_torch(scores, precompute_S_arr, sigma), self.dNDCG_torch(scores, relevances_tensor, iDCG)).sum(dim=1)


    def compute_loss_numpy(self, scores, relevances, iDCG, sigma):
        precompute_S_arr = s_array(relevances, sigma)
        relevances_tensor = self.make_relevance_numinator_numpy(relevances)
        return  np.multiply(self.ranknet_cost_numpy(scores, precompute_S_arr, sigma), self.dNDCG_numpy(scores, relevances_tensor, iDCG))


    def make_relevance_numinator_numpy(self, relevances, exponent=False, add_one=False):
        num_items = relevances.shape[0]
        if add_one:
            relevances += 1
        if exponent:
            pow_relevances = np.power(relevances, 2) - 1
            return np.stack([pow_relevances for _ in range(num_items)])
        else:
            return np.stack([relevances for _ in range(num_items)])

    def make_relevance_numinator_torch(self, relevances, exponent=False, add_one=False):
        num_items = relevances.size()[0]
        if add_one:
            relevances += 1
        if exponent:
            pow_relevances = torch.pow(relevances, 2) - 1
            return torch.stack([pow_relevances for _ in range(num_items)])
        else:
            return torch.stack([relevances for _ in range(num_items)])

    def dNDCG_torch(self, scores, relevances_tensor, iDCG):
        num_items = relevances_tensor.size()[0]
        rank_values = torch_rank_1d(scores).float()
        denominator = 1/torch.log(rank_values + 1)
        rank_denominator_tensor = torch.stack([denominator for _ in range(num_items)])

        return (torch.mul(relevances_tensor, rank_denominator_tensor) - torch.mul(relevances_tensor, rank_denominator_tensor.T))/iDCG


    def dNDCG_numpy(self, scores, relevances_array, iDCG):
        num_items = relevances.shape[0]
        rank_values = numpy_rank_1d(scores).float()
        denominator = 1/np.log(rank_values + 1)
        rank_denominator_tensor = np.stack([denominator for _ in range(num_items)])

        return (np.multiply(relevances_array, rank_denominator_tensor) - np.multiply(relevances_array, rank_denominator_tensor.T))/iDCG


    def ranknet_cost_torch(self, scores, precompute_S_arr, sigma):
        score_unsqueezed = scores.unsqueeze(0)
        score_diff = score_unsqueezed - score_unsqueezed.T
        return precompute_S_arr * score_diff + torch.log(1 + torch.exp(-sigma * score_diff))


    def ranknet_cost_numpy(self, scores, precompute_S_arr, sigma):
        score_unsqueezed = scores.unsqueeze()
        score_diff = score_unsqueezed - score_unsqueezed.T
        return precompute_S_arr * score_diff + np.log(1 + np.exp(-sigma * score_diff))

def s_array(vals, sigma):
    return 0.5*sigma - np.array([[s_value(v1, v2) for v2 in vals] for v1 in vals])*(0.5*sigma)

def s_tensor(vals, sigma):
    return 0.5*sigma - torch.FloatTensor([[s_value(v1, v2) for v2 in vals] for v1 in vals])*(0.5*sigma)

def s_value(v1, v2):
    """calculates S value:
        - v1 > v2: S = 1
        - v1 < v2: S = -1
        - v1 = v2: S = 0
    """
    if v1 > v2:
        return 1
    elif v2 > v1:
        return -1
    else:
        return 0


def torch_rank_1d(values):
    """ Returns tensor with on each index the rank of the value form high to low,
    ranking starts at 1
    ranking is in descending order, e.g:
        values = [6,5,7,8,1]
        ranking = [3,2,4,5,1]

    Todo test this
    """
    return torch.argsort(torch.argsort(values, descending=True)) + 1


def numpy_rank_1d(values):
    """ Returns tensor with on each index the rank of the value form high to low,
    ranking starts at 1
    ranking is in descending order, e.g:
        values = [6,5,7,8,1]
        ranking = [3,2,4,5,1]

    Todo: test this
    """
    return numpy.argsort(numpy.argsort(values, descending=True)) + 1
