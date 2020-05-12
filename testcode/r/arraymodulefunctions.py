class ArrayModuleFunctions:

    """ Short class to redirect function calls to either pytorch or numpy.

    raison d'etre: numpy and pytorch have broadly the smae functions, yet sometimes with different names.

    Returns
    -------
    function redirect class
        Description of returned object.

    """
    def __init__(self, module):
        if module == "torch":
            self.mul = self.torch_mul
            self.power = self.torch_pow
            # raise NotImplementedError()
        elif module == 'numpy':
            raise NotImplementedError()
        else:
            ValueError(f"incorrect module specified, expected 'torch' of 'numpy', got {module}")




def torch_mul(a, b):
    return torch.mul(a, b)


def torch_pow(input_, exponent):
    return torch.pow(input_, exponent)


def torch_log(input_):
    return torch.log(input_)


def torch_ones(input_):
    return torch.ones(input_)

def torch_unsqueeze(input_, dim):
    return torch.unsqueeze(input_, dim)


def numpy_mul(a, b):
    return numpy.multiply(a, b)


def numpy_pow(input_, exponent):
    return np.power(input_, exponent)


def numpy_log(input_):
    return np.log(input_)


def numpy_ones(input_):
    return np.ones(input_)


def numpy_unsqueeze()




#
# def numpy_rank_1d(values):
#     """ Returns tensor with on each index the rank of the value form high to low,
#     ranking starts at 1
#     ranking is in descending order, e.g:
#         values = [6,5,7,8,1]
#         ranking = [3,2,4,5,1]
#
#     Todo: test this
#     """
#     return numpy.argsort(numpy.argsort(values, descending=True)) + 1
#
