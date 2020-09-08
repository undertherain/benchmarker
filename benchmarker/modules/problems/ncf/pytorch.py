# This class is from https://github.com/mlperf.
# https://github.com/mlperf/training/blob/master/recommendation/pytorch/ncf.py

import numpy as np
import torch
import torch.nn as nn

from ..helpers_torch import Recommender


class NeuMF(nn.Module):
    def __init__(
        self, nb_users, nb_items, mf_dim, mf_reg, mlp_layer_sizes, mlp_layer_regs
    ):
        if len(mlp_layer_sizes) != len(mlp_layer_regs):
            raise RuntimeError("u dummy, layer_sizes != layer_regs!")
        if mlp_layer_sizes[0] % 2 != 0:
            raise RuntimeError("u dummy, mlp_layer_sizes[0] % 2 != 0")
        super(NeuMF, self).__init__()
        nb_mlp_layers = len(mlp_layer_sizes)

        self.mf_user_embed = nn.Embedding(nb_users, mf_dim)
        self.mf_item_embed = nn.Embedding(nb_items, mf_dim)
        self.mlp_user_embed = nn.Embedding(nb_users, mlp_layer_sizes[0] // 2)
        self.mlp_item_embed = nn.Embedding(nb_items, mlp_layer_sizes[0] // 2)

        self.mlp = nn.ModuleList()
        for i in range(1, nb_mlp_layers):
            self.mlp.extend(
                [nn.Linear(mlp_layer_sizes[i - 1], mlp_layer_sizes[i])]
            )  # noqa: E501

        self.final = nn.Linear(mlp_layer_sizes[-1] + mf_dim, 1)

        self.mf_user_embed.weight.data.normal_(0.0, 0.01)
        self.mf_item_embed.weight.data.normal_(0.0, 0.01)
        self.mlp_user_embed.weight.data.normal_(0.0, 0.01)
        self.mlp_item_embed.weight.data.normal_(0.0, 0.01)

        def golorot_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            layer.weight.data.uniform_(-limit, limit)

        def lecunn_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features  # noqa: F841, E501
            limit = np.sqrt(3.0 / fan_in)
            layer.weight.data.uniform_(-limit, limit)

        for layer in self.mlp:
            if type(layer) != nn.Linear:
                continue
            golorot_uniform(layer)
        lecunn_uniform(self.final)

    def forward(self, data):
        user, item = data
        # torch.reshape(data, (-1,))
        xmfu = self.mf_user_embed(user)
        xmfi = self.mf_item_embed(item)
        xmf = xmfu * xmfi

        xmlpu = self.mlp_user_embed(user)
        xmlpi = self.mlp_item_embed(item)
        xmlp = torch.cat((xmlpu, xmlpi), dim=1)
        for i, layer in enumerate(self.mlp):
            xmlp = layer(xmlp)
            xmlp = nn.functional.relu(xmlp)

        x = torch.cat((xmf, xmlp), dim=1)
        x = self.final(x)
        # Moved this block and an optional sigmaid=False argument to
        # helpers_torch.py::RecommenderInference()
        #
        # if sigmoid:
        #     x = torch.sigmoid(x)
        return x


def get_kernel(params, unparsed_args=None):
    net = NeuMF(
        nb_users=args.nb_users,
        nb_items=args.nb_items,
        mf_dim=args.factors,
        mf_reg=0.0,
        mlp_layer_sizes=args.layers,
        mlp_layer_regs=[0.0 for i in args.layers],
    )
    return Recommender(params, net)
