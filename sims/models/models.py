import networkx  as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import sklearn.datasets

from sklearn.manifold import TSNE

from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler

from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
torch.set_printoptions(sci_mode=False)


class GCN_Cluster(nn.Module):
    def __init__(self,node_feature_dim, num_communities,layers = 1,dropout = .5) -> None:
        super().__init__()
        self.layers = layers
        self.gcn = nn.Linear(node_feature_dim, node_feature_dim if self.layers > 1 else num_communities)
        self.gcn2 = nn.Linear(node_feature_dim, num_communities)
        self.skip1 = nn.Linear(node_feature_dim, node_feature_dim if self.layers > 1 else num_communities,bias = False)
        self.skip2 = nn.Linear(node_feature_dim, num_communities,bias = False )

        self.act = nn.SELU()
        self.drop = nn.Dropout(dropout)
        self.num_communities = num_communities
        self.node_feature_dim = node_feature_dim
        # self.softmax = nn.Softmax(dim=1) 

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    def forward(self, x_0, A,temp = 1):
        d = torch.diag(torch.pow(torch.sum(A,dim=1),-.5))
        tilde_A = (d@A  @ d) 
        x = tilde_A @ x_0

        x = self.gcn(x) + self.skip1(x_0)

        x = self.act(x)
        x = self.drop(x)
        if self.layers > 1:
            x = tilde_A @ x
            x = self.gcn2(x) + self.skip2(x_0)
    
            x = self.act(x)
            x = self.drop(x)
        x = F.softmax(x,dim=1)
        return x
    
    def produce_gcc_embeddings(self,x_0,A):
        d = torch.diag(torch.pow(torch.sum(A,dim=1),-.5))
        tilde_A = (d@A  @ d) 
        x = tilde_A @ x_0

        x = self.gcn(x) + self.skip1(x_0)

        x = self.act(x)
        if self.layers > 1:
            x = tilde_A @ x
            x = self.gcn2(x) + self.skip2(x_0)
            x = self.act(x)
        #get labels 
        labels  = F.softmax(x,dim=1).argmax(dim=1).detach().cpu().numpy()
        return x,labels
    
    def visualize_gcn_embeddigns(self,x_0,A):
        x,labels = self.produce_gcc_embeddings(x_0,A)
        #use tsne for visualization
        x = TSNE(n_components=2).fit_transform(x.detach().cpu().numpy())

        plt.figure(figsize=(10,10))
        plt.scatter(x[:,0],x[:,1],c=labels)
        plt.show()