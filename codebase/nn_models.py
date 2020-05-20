import torch
import numpy as np
from torch import nn

# ExodiaNet
# De hele eindpresentatie heeft zacht yu-gi-oh muziek op de achtergrond.
# Add memes here

class ModelWrapper:
    """Wrapper class for the model allowing for parallel model initialization"""

    def __init__(self, in_size, hyperparameters):
        random_split = hyperparameters["split_on_random_bool"]

        # DEPRECATED
        prop_split = None

        decay=0.01

        # if random_split and prop_split:
        #     # [rand_bool, prop_known] = [[0, 1], [0, 1]]
        #     self.models = [[], []]
        #     self.forward = self.forward_both

        if random_split:
            self.models = [ExodiaNet(in_size, hyperparameters),
                           ExodiaNet(in_size, hyperparameters)]
            self.forward = self.forward_rand
            self.step = self.split_step
            self.optimizers = [torch.optim.AdamW(self.models[0].parameters(), lr=hyperparameters['learning_rate'], weight_decay=decay),
                               torch.optim.AdamW(self.models[1].parameters(), lr=hyperparameters['learning_rate'], weight_decay=decay)]
            self.clip_grad = self.clip_grad_split

        else:
            self.model = ExodiaNet(in_size, hyperparameters)
            self.forward = self.forward_single
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=hyperparameters['learning_rate'], weight_decay=decay)
            self.step = self.single_step
            self.clip_grad = self.clip_grad_single

    def forward_single(self, _, X):
        return self.model(X)

    def forward_rand(self, rand_bool, X):
        """Forward function for the model when splitting on rand_bool"""
        return self.models[rand_bool](X)

    def single_step(self, _):
        self.optimizer.step()
        self.model.zero_grad()

    def split_step(self, rand_bool):
        self.optimizers[rand_bool].step()
        self.models[rand_bool].zero_grad()

    def clip_grad_split(self, rand_bool):
        torch.nn.utils.clip_grad_norm_(self.models[rand_bool].parameters(), 10)


    def clip_grad_single(self, rand_bool):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)



class ExodiaNet(nn.Module):
    """THE FORBIDDEN ONE"""

    def __init__(self, in_size, hyperparameters):
        super().__init__()

        layer_size = hyperparameters['layer_size']
        layers = hyperparameters['layers']
        attention_layer_idx  = hyperparameters['attention_layer_idx']
        res =  hyperparameters['resnet']
        relu_slope = hyperparameters['relu_slope']

        self.in_size = in_size
        self.out_size = 1
        self.activation = nn.LeakyReLU(relu_slope)
        self.hidden = nn.ModuleList()

        self.hidden.append(nn.Linear(self.in_size, layer_size))
        for i in range(layers - 1):
            if i != attention_layer_idx:
                self.hidden.append(nn.Linear(layer_size, layer_size))
            else:
                self.hidden.append(AttentionLayer(layer_size, layer_size,
                                                  res=res))
        self.hidden.append(nn.Linear(layer_size, self.out_size))

        ################# NEED HE-INIT ###########
        # n_l = input_size * output_size

    def forward(self, x):
        for layer in self.hidden[:-1]:
            x = self.activation(layer(x))
        return self.hidden[-1](x)


class AttentionLayer(nn.Module):

    def __init__(self, in_size, out_size, res=False):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.softmax = nn.Softmax(dim=1)
        self.res = res

        self.weights = nn.Linear(in_size, out_size * 3)

    def forward(self, x):
        key, query, value = self.weights(x).chunk(3, dim=1)
        kq = key @ query.T / np.sqrt(self.out_size)
        kq = self.softmax(kq)

        if self.res:
            return kq @ value + x  # y dos dis work :'(
        return kq @ value
