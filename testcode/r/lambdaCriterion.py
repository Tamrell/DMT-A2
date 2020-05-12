


import numpy as np
import torch


class DeltaNDCG:
    def __init__(self, module):
        pass


    def ddcg(self, scores, relevances):
        """
        # reading material:
        # https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf
        calculates Delta DCG for each pair of documents.
        since DCG is a summation and the difference between an original scoring order
        is determined only by the current contribution of the rank and relevances of a pair,
        this can be quickly calculated by calculating each of the components
        (nominator/relevance and denominator/ranking component) separately
        and combining them appropriately.
        # example (R1 = rank, letters are relevance scores):
        # order:
        #    R1A, R2B, R3C
        # first row: IRM_aa | IRM_ab | IRM_ac
        #    R1A + R1A - R1A - R1A | R1A + R2B - R1B - R2A | R1A + R3C - R1C - R3A
        since the maximum possible dcg is set per subset of data, this is calculated
        separately and divided by as a final step, to save computation.
        :arguments:
            scores, 1d floattensor
            relevances, 1d longtensor (0 to 4 as values)
        :returns:
            diffs, 2d floattensor Delta DCG
        """

        rank_values = torch_rank_1d(scores).float()
        if not len(rank_values.size()):
            rank_values = rank_values.unsqueeze(dim=-1)
        num_items = len(rank_values.tolist())
        denominator = 1/torch.log(rank_values + 1)
        rank_denominator_tensor = torch.stack([denominator for _ in range(num_items)])

        nominator = (torch.pow(torch.ones(relevances.size())*2, relevances) - 1)
        relevances_nominator_tensor = torch.stack([nominator for _ in range(num_items)])

        positive_parts = torch.mul(relevances_nominator_tensor, rank_denominator_tensor)
        positive_combined = positive_parts + torch.transpose(positive_parts, 0, 1)

        negative_parts = torch.mul(torch.transpose(relevances_nominator_tensor, 0, 1), rank_denominator_tensor)
        negative_combined = negative_parts + torch.transpose(negative_parts, 0, 1)

        diffs = positive_combined - negative_combined


        return diffs



class LogisticScoreDifference:
    def __init__(self, sigma):
        raise NotImplementedError()





def torch_rank_1d(values):
    """ Returns tensor with on each index the rank of the value form high to low,
    ranking starts at 1
    ranking is in descending order, e.g:
        values = [6,5,7,8,1]
        ranking = [3,2,4,5,1]

    Todo test this
    """
    return torch.argsort(torch.argsort(values, descending=True)) + 1



if __name__ == '__main__':
    amf = ArrayModuleFunctions("torch")
    a = torch.rand((5,2))
    b = torch.rand((5,2)) *10
    print(amf.mul(a, b))
