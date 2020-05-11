import torch
from torch import nn

# ExodiaNet
# De hele eindpresentatie heeft zacht yu-gi-oh muziek op de achtergrond.
# Add memes here

class ExodiaNet(nn.Module):
    """THE FORBIDDEN ONE"""

    def __init__(self, model_id, in_size, layer_size, layers,
                 attention_layer_idx, res, relu_slope):
        super().__init__()

        self.in_size = in_size
        self.out_size = 1
        self.activation = n.LeakyReLu(relu_slope)
        self.hidden = nn.ModuleList()
        self.model_id = model_id

        self.hidden.append(nn.Linear(self.in_size, layer_size))
        for i in range(layers - 1):
            if i != attention_layer_idx:
                self.hidden.append(nn.Linear(layer_size, layer_size))
            else:
                self.hidden.append(AttentionLayer(layer_size, layer_size,
                                                  res=res))
        self.hidden.append(nn.Linear(layer_size, self.out_size))

    def forward(self, x):
        for layer in self.hidden:
            x = self.activation(layer(x))
        return x


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
        kq = key @ query / torch.sqrt(self.out_size)
        kq = self.softmax(kq)

        if self.res:
            return kq @ value + x  # y dos dis work :'(
        return kq @ value


