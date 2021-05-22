"""
@author: Wei Jiang

基于torch_geometric，models文件用于存放quickGCN所调用的经典GCN模型

目前能够调用的GCN模型有：ChebNet、GGNN、GCN、GAT

file ‘models’ uses to save the GCN models which quickGCNs calls to train based on torch_geometric.

Available models: ChebNet, GGNN, GCN, GAT

Update on 2021/05/22
"""


import torch_geometric.nn as pyg_nn
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, model_params):
        super(GCN, self).__init__()
        # model_params
        feat_dims, hidden_dims, actfunc_type, layer_num, k = model_params
        self.feat_dims = feat_dims
        self.hidden_dims = hidden_dims
        self.actfunc_type = actfunc_type
        self.layer_num = layer_num

        # define GCNconvs
        self.gconvs = []
        for layer in range(self.layer_num):
            if layer==0:
                self.gconv = pyg_nn.GCNConv(self.feat_dims, self.hidden_dims)
                self.gconvs.append(self.gconv)
            else:
                self.gconv = pyg_nn.GCNConv(self.hidden_dims, self.hidden_dims)
                self.gconvs.append(self.gconv)

        # define Linear layer
        self.linear = nn.Linear(self.hidden_dims, 1)
    
    def activate_function(self, actfunc_type):
        if actfunc_type == 'relu':
            act_func = nn.ReLU(True)
        elif actfunc_type == 'sigmoid':
            act_func = nn.Sigmoid()
        elif actfunc_type == 'tanh':
            act_func = nn.Tanh()
        elif actfunc_type == 'leakyReLU':
            act_func = nn.LeakyReLU(True)
        
        return act_func

    def forward(self, data):
        
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        for layer in range(self.layer_num):
            self.gconv = self.gconvs[layer]
            x = self.gconv(x, edge_index)
            act_func = self.activate_function(self.actfunc_type)
            x = act_func(x)

        x = self.linear(x)
        
        return x

    def forward(self, data):
        
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        for layer in range(self.layer_num):
            self.gconv = self.gconvs[layer]
            x = self.gconv(x, edge_index)
            act_func = self.activate_function(self.actfunc_type)
            x = act_func(x)

        x = self.linear(x)
        
        return x

class GAT(nn.Module):
    def __init__(self, model_params):
        super(GAT, self).__init__()
        # model_params
        feat_dims, hidden_dims, actfunc_type, layer_num, k = model_params
        self.feat_dims = feat_dims
        self.hidden_dims = hidden_dims
        self.actfunc_type = actfunc_type
        self.layer_num = layer_num

        # define GATconvs
        self.gconvs = []
        for layer in range(self.layer_num):
            if layer==0:
                self.gconv = pyg_nn.GATConv(self.feat_dims, self.hidden_dims)
                self.gconvs.append(self.gconv)
            else:
                self.gconv = pyg_nn.GATConv(self.hidden_dims, self.hidden_dims)
                self.gconvs.append(self.gconv)

        # define Linear layer
        self.linear = nn.Linear(self.hidden_dims, 1)
    
    def activate_function(self, actfunc_type):
        if actfunc_type == 'relu':
            act_func = nn.ReLU(True)
        elif actfunc_type == 'sigmoid':
            act_func = nn.Sigmoid()
        elif actfunc_type == 'tanh':
            act_func = nn.Tanh()
        elif actfunc_type == 'leakyReLU':
            act_func = nn.LeakyReLU(True)
        
        return act_func

    def forward(self, data):
        
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        for layer in range(self.layer_num):
            self.gconv = self.gconvs[layer]
            x = self.gconv(x, edge_index)
            act_func = self.activate_function(self.actfunc_type)
            x = act_func(x)

        x = self.linear(x)
        
        return x

class GGNN(nn.Module):
    def __init__(self, model_params):
        super(GGNN, self).__init__()
        # model_params
        feat_dims, hidden_dims, actfunc_type, layer_num, k = model_params
        self.feat_dims = feat_dims
        self.hidden_dims = hidden_dims
        self.actfunc_type = actfunc_type
        self.layer_num = layer_num

        # define GGNNconvs
        self.gconv = pyg_nn.GatedGraphConv(out_channels=feat_dims, num_layers=self.layer_num)

        # define Linear layer
        self.linear = nn.Linear(self.hidden_dims, 1)
    
    def activate_function(self, actfunc_type):
        if actfunc_type == 'relu':
            act_func = nn.ReLU(True)
        elif actfunc_type == 'sigmoid':
            act_func = nn.Sigmoid()
        elif actfunc_type == 'tanh':
            act_func = nn.Tanh()
        elif actfunc_type == 'leakyReLU':
            act_func = nn.LeakyReLU(True)
        
        return act_func

    def forward(self, data):
        
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        x = self.gconv(x, edge_index)
        act_func = self.activate_function(self.actfunc_type)
        x = act_func(x)

        x = self.linear(x)
        
        return x

class ChebNet(nn.Module):
    def __init__(self, model_params):
        super(ChebNet, self).__init__()
        
        # model_params
        feat_dims, hidden_dims, actfunc_type, layer_num, k = model_params
        self.feat_dims = feat_dims
        self.hidden_dims = hidden_dims
        self.actfunc_type = actfunc_type
        self.layer_num = layer_num

        # define ChebNetconvs
        self.gconvs = []
        for layer in range(self.layer_num):
            if layer==0:
                self.gconv = pyg_nn.ChebConv(self.feat_dims, self.hidden_dims, K=k)
                self.gconvs.append(self.gconv)
            else:
                self.gconv = pyg_nn.ChebConv(self.hidden_dims, self.hidden_dims, K=k)
                self.gconvs.append(self.gconv)

        # define Linear layer
        self.linear = nn.Linear(self.hidden_dims, 1)
    
    def activate_function(self, actfunc_type):
        if actfunc_type == 'relu':
            act_func = nn.ReLU(True)
        elif actfunc_type == 'sigmoid':
            act_func = nn.Sigmoid()
        elif actfunc_type == 'tanh':
            act_func = nn.Tanh()
        elif actfunc_type == 'leakyReLU':
            act_func = nn.LeakyReLU(True)
        
        return act_func

    def forward(self, data):
        
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        for layer in range(self.layer_num):
            self.gconv = self.gconvs[layer]
            x = self.gconv(x, edge_index)
            act_func = self.activate_function(self.actfunc_type)
            x = act_func(x)

        x = self.linear(x)
        
        return x