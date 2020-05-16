import torch

class lambdaRankCriterion:
    """Criterion:
        calculates lambda-gradients
        calculates NDCG values. (not applicable for gradient, class might be split further, and refactored out further)

    Parameters
    ----------
    exp_ver --> attribute: bool
        True indicates the 2^relevance + 1 is the nominator of NDCG.
        False indicates the version the course is telling us to use
            (here the nominator of DCG is just relevance (in [0,1,5]))
    device --> attribute: str
        torch device used to direct values to specific device.
    sigma --> attribute: float
        sigma to adjust ranknet boundary.
        [tried a bunch of different vlaues at this point and doesnt seem to matter anything]

    Attributes
    ----------
    same as input:
        exp_ver
        device
        sigma
    other attributes:
        -
    """

    def __init__(self, exp_ver, device, sigma):
        self.exp_ver = exp_ver
        self.device = device
        self.sigma = sigma

    def calculate_gradient_and_NDCG(self, y_pred, Y):
        """Calculates combined lambda @ ranknet gradients, as well as NDCG values.

        All these are calculated in one function as this is more efficient w.r.t.
        values co-occuring between the different calculations.

        Parameters
        ----------
        y_pred : tensor [todo check exact shape]
            neural network output, higher values are ranked higher
        Y : tensor, same shape as y_pred
            relevance label in [0,1,5]

        Returns
        -------
        grad : tensor
            lambdarank gradients.
        NDCG_train : 0d tensor
            NDCG of training query
        NDCG_train_at5 : 0d tensor
            NDCG@5 of training query
        """
        rank_order_as_NDCG_denominator = torch.argsort(torch.argsort(Y.squeeze(), descending=True)) + 2
        rank_order_as_NDCG_denominator_tensor = torch.as_tensor(rank_order_as_NDCG_denominator, dtype=float).view(-1, 1).to(self.device)

        ranknet_cost = self.calc_ranknet_gradients(y_pred, Y)
        NDCG_relevance_grade = self.adjust_for_EXP_NDCG_relevance(Y)
        delta_ndcg, maxDCG_elements = self.calc_nDCG_gradients(NDCG_relevance_grade, rank_order_as_NDCG_denominator_tensor)

        NDCG_train, NDCG_train_at5 = self.calc_NDCG(y_pred, NDCG_relevance_grade, maxDCG_elements)
        lambda_update = ranknet_cost * delta_ndcg
        grad = torch.sum(lambda_update, 1, keepdim=True)
        return grad, NDCG_train, NDCG_train_at5

    def calc_NDCG_val(self, y_pred, Y_val):
        """Calculates only NDCG and NDCG@5, implemented separately for
        validation, as that requeres no gradients.


        Parameters
        ----------
        y_pred : tensor [todo: check shape]
            network output
        Y_val : tensor
            original relevance/label/Y

        Returns
        -------
        type
            NDCG and NDCG@5 given prediction scores.

        """
        val_dcg_pred_at5_idx = torch.argsort(y_pred.squeeze(), descending=True)[:5]
        val_dcg_max_at5_idx = torch.argsort(Y_val.squeeze(), descending=True)[:5]
        NDCG_relevance_grade = self.adjust_for_EXP_NDCG_relevance(Y_val)

        val_dcg_pred_elements = NDCG_relevance_grade.squeeze() / torch.log2(torch.argsort(torch.argsort(y_pred.squeeze(), descending=True)).float() + 2)
        val_dcg_max_elements = NDCG_relevance_grade.squeeze() / torch.log2(torch.argsort(torch.argsort(Y_val.squeeze(), descending=True)).float() + 2)

        val_ndcg = (torch.sum(val_dcg_pred_elements)/torch.sum(val_dcg_max_elements)).item()
        val_ndcg_at5 = (torch.sum(val_dcg_pred_elements[val_dcg_pred_at5_idx])/torch.sum(val_dcg_max_elements[val_dcg_max_at5_idx])).item()
        return val_ndcg, val_ndcg_at5

    def calc_NDCG(self, y_pred, relevance_factor, maxDCG_elements):
        """calculate NDCG and NDCG@5.

        uses input-arguments that are already avalailable when calculating
        gradients for faster computation

        Parameters
        ----------
        y_pred : tensor
            network prediction.
        relevance_factor : tensor
            relevance in the format the decided way of DCG needs it.
        maxDCG_elements : tensor
            elements of the NDCG according to ideal ordering (not summed)

        Returns
        -------
        float, float
            NDCG and NDCG@5 of the search_id the output originates from.

        """
        dcg_pred_elements = relevance_factor.squeeze() / torch.log2(torch.argsort(torch.argsort(y_pred.squeeze(), descending=True)).float() + 2)
        idx = torch.argsort(y_pred, descending=True)
        idx_at5 = idx[:5]
        NDCG_train = torch.sum(dcg_pred_elements)/torch.sum(maxDCG_elements)
        NDCG_train_at5 = torch.sum(dcg_pred_elements[idx_at5])/torch.sum(maxDCG_elements[idx_at5])

        return NDCG_train.item(), NDCG_train_at5.item()


    def calc_gain_diff(self, relevance_factor):
        """calculates nominator of DCG, independent of NDCG type.

        the DCG type is already processed in the input

        Parameters
        ----------
        relevance_factor : tensor
            relevance in the format the decided way of DCG needs it.

        Returns
        -------
        tensor
            pairwise difference in nominator of DCG for all results in query.

        """
        return relevance_factor - relevance_factor.t()


    def calc_nDCG_gradients(self, relevance_factor, rank_order_as_NDCG_denominator_tensor):
        """calculate deltaNDCG elements.

        primary functionality of this method is to calculate deltaNDCG
        in the process of this the elements for the maximum NDCG of the
        search_id is calculated, returning this as well allows for an
        optimization in the calculation of train NDCG (which might not actually
        be necessary, but if we want to log it this is the way to go w.r.t.
        calculation)

        Parameters
        ----------
        relevance_factor : tensor
            relevance in the format the decided way of DCG needs it.
        rank_order_as_NDCG_denominator_tensor : tensor [todo: check whether [Nx1] or [1xN] (does not matter for functionality as absolute value is taken)]
            tensor with at each element the ranking of the corresponding element
            (starting from 2) according to the TRUE LABELS.

        Returns
        -------
        type
            Description of returned object.

        """
        denominator_dcg_max_DCG_elements = 1 / torch.log2(rank_order_as_NDCG_denominator_tensor)
        maxDCG_elements = relevance_factor.squeeze() * denominator_dcg_max_DCG_elements
        maxDCG = torch.sum(maxDCG_elements).to(self.device)

        N = 1.0 / maxDCG
        gain_diff = self.calc_gain_diff(relevance_factor)
        decay_diff = 1.0 * denominator_dcg_max_DCG_elements - 1.0 * denominator_dcg_max_DCG_elements.t()

        delta_ndcg = torch.abs(N * gain_diff * decay_diff)
        return delta_ndcg, maxDCG_elements

    def calc_ranknet_gradients(self, y_pred, Y):
        """Calculates gradients for the ranknet component.

        Parameters
        ----------
        y_pred : tensor [todo:check shape]
            network output
        Y : tensor
            original relevance/label/Y value (in [0,1,5]).

        Returns
        -------
        tensor
            ranknet cost gradient.

        """
        rel_diff = Y - Y.t()
        pos_pairs = (rel_diff > 0).float()
        neg_pairs = (rel_diff < 0).float()
        # Hier aanpassen voor @5? - A
        # volgens mij wel, 1 van de twee moet dan exclusief over alleen de
        # beste 5 gaan, als je beide doet heb je helemaal geen gradients voor
        # een grote subset. welke het is zou je in princiepe kunnen checken
        # door 1 te wijzigen, te runnen en als het niet werkt dan te transposen

        Sij = pos_pairs - neg_pairs
        pos_pairs_score_diff = 1.0 + torch.exp(self.sigma* (y_pred - y_pred.t()))
        return self.sigma * (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff)

    def adjust_for_EXP_NDCG_relevance(self, relevance_value):
        """Adjusts real relevances/labels/Y to be the relevance formulation

        corresponding to NDCG type. --> see class description

        Parameters
        ----------
        relevance_value : tensor
            relevance/label/y in [0,1,5]

        Returns
        -------
        tensor
            either original relevance or the 2^relevance + 1, depending on
            NDCG version
        """
        if self.exp_ver:
            return torch.pow(2, relevance_value) + 1
        else:
            return relevance_value
