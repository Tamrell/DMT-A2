


import numpy as np
import torch


class DeltaNDCG:
    def __init__(self, module):
        if module == "numpy":
            self.compute_loss = self.compute_loss_numpy
        elif module == 'pytorch':
            self.compute_loss = self.compute_loss_torch

    @staticmethod
    def compute_loss_torch(scores, relevances_tensor, iDCG, precompute_S_arr, sigma):
        return  torch.mul(self.ranknet_cost_torch(scores, precompute_S_arr, sigma), dNDCG_torch(scores, relevances_tensor, iDCG))


    @staticmethod
    def compute_loss_numpy(scores, relevances_tensor, iDCG, precompute_S_arr, sigma):
        return  np.multiply(self.ranknet_cost_numpy(scores, precompute_S_arr, sigma), dNDCG_numpy(scores, relevances_tensor, iDCG))


    @staticmethod
    def make_relevance_numinator(relevances, exponent=True, add_one=True):
        num_items = relevances.shape[0]
        if add_one:
            relevances += 1
        if exponent:
            return np.power(relevances, 2) - 1
        else:
            return relevances


    @staticmethod
    def dNDCG_torch(scores, relevances_tensor, iDCG):
        num_items = relevances.size[0]
        rank_values = torch_rank_1d(scores).float()
        denominator = 1/torch.log(rank_values + 1)
        rank_denominator_tensor = torch.stack([denominator for _ in range(num_items)])

        return (torch.mul(relevances_tensor, rank_denominator_tensor) - torch.mul(relevances_tensor, rank_denominator_tensor.T))/iDCG


    @staticmethod
    def dNDCG_numpy(scores, relevances_array, iDCG):
        num_items = relevances.shape[0]
        rank_values = numpy_rank_1d(scores).float()
        denominator = 1/np.log(rank_values + 1)
        rank_denominator_tensor = np.stack([denominator for _ in range(num_items)])

        return (np.multiply(relevances_array, rank_denominator_tensor) - np.multiply(relevances_array, rank_denominator_tensor.T))/iDCG


    @staticmethod
    def ranknet_cost_torch(scores, precompute_S_arr, sigma):
        score_unsqueezed = scores.unsqueeze()
        score_diff = score_unsqueezed - score_unsqueezed.T
        return precompute_S_arr * score_diff + torch.log(1 + torch.exp(-sigma * score_diff))


    @staticmethod
    def ranknet_cost_numpy(scores, precompute_S_arr, sigma):
        score_unsqueezed = scores.unsqueeze()
        score_diff = score_unsqueezed - score_unsqueezed.T
        return precompute_S_arr * score_diff + np.log(1 + np.exp(-sigma * score_diff))



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
