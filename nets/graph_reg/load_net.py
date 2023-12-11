"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.graph_reg.gcn_net import GCNNet
from nets.graph_reg.gin_net import GINNet
from nets.graph_reg.unionsnn_net import UnionSNNNet


def GCN(net_params):
    return GCNNet(net_params)


def GIN(net_params):
    return GINNet(net_params)


def UnionSNN(net_params):
    return UnionSNNNet(net_params)


def gnn_model(MODEL_NAME, net_params):
    models = {
        'GCN': GCN,
        'GIN': GIN,
        'UnionSNN': UnionSNN,
    }

    model = models[MODEL_NAME](net_params)
    model.name = MODEL_NAME
        
    return model
