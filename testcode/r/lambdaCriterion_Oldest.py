import numpy as np
import torch

torch.set_printoptions(threshold=10000)


class DeltaNDCG:
    def __init__(self, module):
        if module == "numpy":
            self.compute_loss = self.compute_loss_numpy
        elif module == 'pytorch':
            self.compute_loss = self.compute_loss_torch

    def compute_loss_torch(self, scores, relevances,  iDCG, sigma, device):
        precompute_S_arr = s_tensor(relevances, sigma).to(device)
        relevances_tensor = self.make_relevance_nominator_torch(relevances)
        # print("precompute_S_arr", precompute_S_arr)
        # print("sigma", sigma)
        # print("scores", scores)
        # print("relevances_tensor", relevances_tensor)
        # print("iDCG", iDCG)
        rnet_cost = self.ranknet_cost_torch(scores, precompute_S_arr, sigma)
        dndcg, denominator = self.dNDCG_torch(scores, relevances_tensor, iDCG)
        # print("rnet_cost", rnet_cost)
        # print("dndcg", dndcg) # not finished
        # input()
        return  torch.mul(rnet_cost, dndcg).sum(dim=1), denominator


    def compute_loss_numpy(self, scores, relevances, iDCG, sigma):
        precompute_S_arr = s_array(relevances, sigma)
        relevances_tensor = self.make_relevance_nominator_numpy(relevances)
        return  np.multiply(self.ranknet_cost_numpy(scores, precompute_S_arr, sigma), self.dNDCG_numpy(scores, relevances_tensor, iDCG))


    def make_relevance_nominator_numpy(self, relevances, exponent=False, add_one=False):
        num_items = relevances.shape[0]
        if add_one:
            relevances += 1
        if exponent:
            pow_relevances = np.power(relevances, 2) - 1
            return np.stack([pow_relevances for _ in range(num_items)])
        else:
            return np.stack([relevances for _ in range(num_items)])

    def make_relevance_nominator_torch(self, relevances, exponent=False, add_one=False):
        num_items = relevances.size()[0]
        if add_one:
            relevances += 1
        if exponent:
            pow_relevances = torch.pow(relevances, 2) - 1
            return torch.stack([pow_relevances for _ in range(num_items)])
        else:
            return torch.stack([relevances for _ in range(num_items)])

    def dNDCG_torch(self, scores, relevances_tensor, iDCG):
        # print("relevances_tensor.size()", relevances_tensor.size())
        num_items = relevances_tensor.size()[0]
        rank_values = torch_rank_1d(scores).float()
        denominator = 1./torch.log2(rank_values + 1)
        rank_denominator_tensor = torch.stack([denominator for _ in range(num_items)])

        relevances_tensor = relevances_tensor.squeeze()

        direct_delta_denominator = torch.abs(rank_denominator_tensor - rank_denominator_tensor.T)
        direct_delta_nominator = relevances_tensor - relevances_tensor.T

        # print(direct_delta_denominator)
        # print("direct_delta_nominator")
        # print(direct_delta_nominator)
        # #
        # input()

        # original = torch.mul(relevances_tensor, rank_denominator_tensor)
        # after_swap = torch.mul(relevances_tensor, rank_denominator_tensor.T)

        # print("rank_denominator_tensor", rank_denominator_tensor)
        # print("rank_denominator_tensor.size()", rank_denominator_tensor.size())
        # print("rank_values", rank_values)

        # return (torch.mul(relevances_tensor, rank_denominator_tensor) - torch.mul(relevances_tensor, rank_denominator_tensor.T))/iDCG

        # return ((original + original.T) - (after_swap + after_swap.T))/iDCG, denominator
        dNDCG = torch.mul(direct_delta_denominator, direct_delta_nominator)/iDCG
        if torch.sum(torch.isnan(dNDCG)):
            print("dNDCG has nans")
            print("iDCG", iDCG)
            print("direct_delta_denominator")
            print(direct_delta_denominator)
            print("direct_delta_nominator")
            print(direct_delta_nominator)
            exit()

        return dNDCG, denominator


    def dNDCG_numpy(self, scores, relevances_array, iDCG):
        """ IMPLEMENTATION CHANGED RE-IMPLEMENT BASED ON TORCH VERSION ABOVE @R

        """

        num_items = relevances.shape[0]
        rank_values = numpy_rank_1d(scores).float()
        denominator = 1/np.log2(rank_values + 1)
        rank_denominator_tensor = np.stack([denominator for _ in range(num_items)])

        return (np.multiply(relevances_array, rank_denominator_tensor) - np.multiply(relevances_array, rank_denominator_tensor.T))/iDCG


    def ranknet_cost_torch(self, scores, precompute_S_arr, sigma):
        score_unsqueezed = scores
        score_diff = score_unsqueezed - score_unsqueezed.T

        score_dif_norm = torch.norm(score_diff)

        log_exp_part = torch.log(1 + torch.exp(-sigma * (score_diff/score_dif_norm)))
        RNCost = precompute_S_arr * score_diff/score_dif_norm + log_exp_part


        if torch.sum(torch.isnan(RNCost)):
            print("RNCost has nans")
            print("precompute_S_arr")
            print(precompute_S_arr)
            print("score_diff")
            print(score_diff)
            print("log_exp_part")
            print(log_exp_part)
            print("scores", scores)
            exit()

        return RNCost


    def ranknet_cost_numpy(self, scores, precompute_S_arr, sigma):
        """ Todo: check shapes

        """
        score_unsqueezed = scores.unsqueeze()
        print("score_unsqueezed", score_unsqueezed)
        score_diff = score_unsqueezed - score_unsqueezed.T
        print("score_diff", score_diff)
        return precompute_S_arr * score_diff + np.log(1 + np.exp(-sigma * score_diff))

def s_array(vals, sigma):
    return 0.5*sigma - np.array([[s_value(v1, v2) for v2 in vals] for v1 in vals])*(0.5*sigma)

def s_tensor(vals, sigma):
    precompute_S_arr = 0.5*sigma - (vals-vals.T).sign()*(0.5*sigma)
    if torch.sum(torch.isnan(precompute_S_arr)):
        print("s_tensor has nans")
        print("precompute_S_arr")
        print(precompute_S_arr)
        print("(vals-vals.T)")
        print(vals-vals.T)
        print("(vals-vals.T).sign()")
        print((vals-vals.T).sign())
        exit()
    return precompute_S_arr
    # return 0.5*sigma - torch.FloatTensor([[s_value(v1, v2) for v2 in vals] for v1 in vals])*(0.5*sigma)


def torch_rank_1d(values):
    """ Returns tensor with on each index the rank of the value form high to low,
    ranking starts at 1
    ranking is in descending order, e.g:
        values = [6,5,7,8,1]
        ranking = [3,2,4,5,1]

    Todo test this
    """
    values = values.squeeze()
    # print("got scores", values)
    # print("argsort(scores)", torch.argsort(values, descending=True))
    # print("argsort(argsort(scores))", torch.argsort(torch.argsort(values, descending=True)))
    return torch.argsort(torch.argsort(values, descending=True)) + 1


def numpy_rank_1d(values):
    """ Returns tensor with on each index the rank of the value form high to low,
    ranking starts at 1
    ranking is in descending order, e.g:
        values = [6,5,7,8,1]
        ranking = [3,2,4,5,1]

    Todo: test this, possibly needs squeeze
    """
    return numpy.argsort(numpy.argsort(values, descending=True)) + 1
