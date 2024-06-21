from abc import ABC, abstractmethod
from .simulations import GenerateStochasticBlockModelWithFeatures
import numpy as np 
import matplotlib.pyplot as plt
import graph_tool.all as gt
import copy
import networkx as nx
# import matplotlib.colors as 
from math import sqrt
import torch



class Experiment: 
    def __init__(self) -> None:
        pass
    @abstractmethod
    def visualize(self):
        pass
    @abstractmethod
    def change(self):
        pass
    @abstractmethod
    def make_proportion_matrix(self,pi):
        pass 
    @abstractmethod
    def resample_graph(self): 
        pass
    @abstractmethod
    def make_data(self ):
        pass 
    
class BaseExperiment(Experiment):
    def __init__(self, n,num_edges,pi,prop_mat,in_degree,out_degree,num_features = 62,degree_law = 'power') -> None:
        super().__init__()
        self.num_features = num_features

        self.n = n
        self.num_edges = num_edges
        self.in_degree = in_degree
        self.out_degree = out_degree
        self.pi = pi
        self.make_proportion_matrix(len(pi))

        if degree_law == 'power': 
            self.degrees = np.random.uniform(low =2**2 , high=(2**2)**2, size=n)
        self.G = GenerateStochasticBlockModelWithFeatures(n,num_edges,pi,prop_mat=prop_mat,feature_dim = num_features,feature_center_distance =3,out_degs=self.degrees, centers = None )
        self.convert_to_nx()

    def visualize_degree(self):
        plt.hist(self.G.graph.degree_property_map("total").a, bins=100)
        plt.show()  
        
    def convert_to_nx(self):
        # Copy node properties
        g = self.G.graph
        nx_graph = nx.Graph(directed = False)

        for v,node_label in zip(g.vertices(),self.G.graph_memberships):
            node_id = int(v)

            nx_graph.add_node(node_id, label=node_label)

        # Copy edge properties

        for e in g.edges():
            source = int(e.source())
            target = int(e.target())

            nx_graph.add_edge(source, target)
        self.nx_graph = nx_graph

    def visualize_network(self, file_name='graph.pdf', community_colors=None):
            self.convert_to_nx()

            cmap = plt.cm.get_cmap("tab10")  # Choose a colormap (e.g., "tab10")
            deg = np.array(self.G.graph_memberships)
            pos = nx.spring_layout(self.nx_graph)
            d = dict(self.nx_graph.degree)

            if community_colors is not None:
                nx.draw(self.nx_graph, pos=pos, node_color=community_colors, node_size=[v * 1.5 for v in d.values()],
                        width=0.1, alpha=0.7, linewidths=1.2)
            else:
                nx.draw(self.nx_graph, pos=pos, node_color=deg, cmap=cmap, node_size=[v * 1.5 for v in d.values()],
                        width=0.1, alpha=0.7, linewidths=1.2)

            plt.savefig(file_name) 
            

    def make_proportion_matrix(self, num_communities):
        #make a diaganoal matrix with in degree on the diaganoals and out degree everywhere else 
        self.prop_mat = np.empty((num_communities,num_communities))
        self.prop_mat.fill(self.out_degree)
        
        np.fill_diagonal(self.prop_mat,self.in_degree)

    def change(self):
        pass
    def resample_graph(self): 
        self.G.node_features_centers = np.array(self.G.node_features_centers)
        noise = np.random.normal(loc=0, scale=0.1, size=(self.G.node_features_centers.shape[0], self.G.node_features_centers.shape[1]))
        self.G = GenerateStochasticBlockModelWithFeatures(self.n,self.num_edges,self.pi,prop_mat=self.prop_mat,feature_dim = self.num_features,
                                                        feature_center_distance =0,out_degs=self.degrees, 
                                                        centers = np.array(self.G.node_features_centers )+ noise , sbm = copy.deepcopy(self.G))
    def make_data(self,device):

        A = gt.adjacency(self.G.graph).todense()
        
        A = torch.tensor(A,requires_grad=False).float().to(device)
        x = torch.tensor(self.G.node_features1,requires_grad=False).float().to(device)
        edge_index = torch.tensor(self.G.graph.get_edges().tolist(),requires_grad=False).T.to(device)
        return x,edge_index,A
    

class ShiftCovariatesGradual(BaseExperiment):
    def __init__(self,  n, num_edges, pi, prop_mat,in_degree,out_degree, num_features=62, degree_law='power') -> None:
        super().__init__( n, num_edges, pi, prop_mat,in_degree,out_degree, num_features, degree_law)

    def change(self,step, change_type = 'linear',noise = True):
        if change_type == 'linear': 
            self.G.node_features_centers = np.array(self.G.node_features_centers)
            self.G.node_features_centers = self.G.node_features_centers * step + np.random.uniform(low=0, high=1, size=(self.G.node_features_centers.shape[0], self.G.node_features_centers.shape[1])) if noise else 0 
            # print(f"==>> {self.G.node_features_centers=}")
            # print(f"==>> {self.G.node_features_centers=}")
            self.G =GenerateStochasticBlockModelWithFeatures(self.n,self.num_edges,self.pi,prop_mat=self.prop_mat,feature_dim = self.num_features,feature_center_distance =1/10,out_degs=self.degrees, centers = self.G.node_features_centers, sbm = copy.deepcopy(self.G))
        else: 
            raise Exception('Not Implemented')

class ShiftCovariatesAbrupt(BaseExperiment):
    def __init__(self,  n, num_edges, pi, prop_mat,in_degree,out_degree, num_features=62, degree_law='power') -> None:
        super().__init__( n, num_edges, pi, prop_mat,in_degree,out_degree, num_features, degree_law)


    def change(self,gravity = 10):
        self.G.node_features_centers = np.array(self.G.node_features_centers)
        self.G.node_features_centers  = self.G.node_features_centers  + gravity* np.random.normal(loc=0, scale=1, size=(self.G.node_features_centers.shape[0], self.G.node_features_centers.shape[1]))
        self.G =GenerateStochasticBlockModelWithFeatures(self.n,self.num_edges,self.pi,prop_mat=self.prop_mat,feature_dim = self.num_features,feature_center_distance =1/10,out_degs=self.degrees, centers = self.G.node_features_centers, sbm = copy.deepcopy(self.G))

class SplitCommunity(BaseExperiment):
    def __init__(self,  n, num_edges, pi, prop_mat,in_degree,out_degree, num_features=62, degree_law='power') -> None:
        super().__init__( n, num_edges, pi, prop_mat, in_degree,out_degree,num_features, degree_law)
    
    
            
    def change(self):

        n_coms= self.prop_mat.shape[0] +1
        self.make_proportion_matrix(n_coms)

        self.pi = np.concatenate([self.pi[:-1],[.5,.5]])
        
        new_centers = np.random.normal(loc=0, scale=0.1, size=(1, self.num_features))
        self.G.node_features_centers = np.concatenate([self.G.node_features_centers,new_centers])

        
        assert self.pi.shape[0] == self.G.node_features_centers.shape[0]
        if self.prop_mat.shape[0] != len(self.pi) or self.prop_mat.shape[1] != len(self.pi):
               raise ValueError("prop_mat must be k x k; k = len(pi1) + len(pi2)")
        
        self.G = GenerateStochasticBlockModelWithFeatures(self.n,self.num_edges,
                                                         self.pi,
                                                         prop_mat=self.prop_mat,
                                                         feature_dim = self.num_features,
                                                         feature_center_distance =1/10,
                                                         out_degs=self.degrees,
                                                         centers = self.G.node_features_centers,
                                                         sbm = copy.deepcopy(self.G))
   


class ChangeAttribute(BaseExperiment):
    def __init__(self,  n, num_edges, pi, prop_mat,in_degree,out_degree, num_features=62, degree_law='power') -> None:
        super().__init__( n, num_edges, pi, prop_mat, in_degree,out_degree,num_features, degree_law)
    
    
            
    def change(self, attribute, new_value):
            if hasattr(self, attribute):
                setattr(self, attribute, new_value)
                print(f"Attribute '{attribute}' changed to {new_value}.")
            else:
                print(f"Attribute '{attribute}' does not exist.")


 

        



class MergeCommunity(BaseExperiment):
    def __init__(self,  n, num_edges, pi, prop_mat,in_degree,out_degree, num_features=62, degree_law='power') -> None:
        super().__init__( n, num_edges, pi, prop_mat, in_degree,out_degree,num_features, degree_law)
    
            
     
    def change(self):
        n_coms= self.prop_mat.shape[0] -1 
        self.make_proportion_matrix(n_coms)
        self.pi = self.pi[:-1]
        # self.pi[-1] *= 2 
        merged_centers = np.mean(self.G.node_features_centers[-2:],axis=0)
        self.G.node_features_centers = np.concatenate([self.G.node_features_centers[:-2],[merged_centers]])

        assert self.prop_mat.shape[0] == self.G.node_features_centers.shape[0]
        
        self.G =GenerateStochasticBlockModelWithFeatures(self.n,self.num_edges,
                                                         self.pi,
                                                         prop_mat=self.prop_mat,
                                                         feature_dim = self.num_features,
                                                         feature_center_distance =1/10,
                                                         out_degs=self.degrees,
                                                         centers = self.G.node_features_centers,
                                                         sbm = copy.deepcopy(self.G))
        
        


        

class NewCommunity(BaseExperiment):
    def __init__(self,  n, num_edges, pi, prop_mat,in_degree,out_degree, num_features=62, degree_law='power') -> None:
        super().__init__( n, num_edges, pi, prop_mat, in_degree,out_degree,num_features, degree_law)
     
    def change(self):
        n_per_community = int(self.n / self.prop_mat.shape[0])
        self.n +=  n_per_community
        new_community = np.random.normal(loc=0, scale=0.1, size=(1, self.num_features))
        
        
        n_coms= self.prop_mat.shape[0] + 1
        self.make_proportion_matrix(n_coms)
        self.G.node_features_centers = np.concatenate([self.G.node_features_centers,new_community])
        self.pi = np.concatenate([self.pi,[1]])
        
     
        assert self.prop_mat.shape[0] == self.G.node_features_centers.shape[0]
        
        self.G =GenerateStochasticBlockModelWithFeatures(self.n,self.num_edges,
                                                         self.pi,
                                                         prop_mat=self.prop_mat,
                                                         feature_dim = self.num_features,
                                                         feature_center_distance =1/10,
                                                         out_degs=self.degrees,
                                                         centers = self.G.node_features_centers,
                                                         sbm = copy.deepcopy(self.G))
        
class NoChange(BaseExperiment):
    def __init__(self,  n, num_edges, pi, prop_mat,in_degree,out_degree, num_features=62, degree_law='power') -> None:
        super().__init__( n, num_edges, pi, prop_mat, in_degree,out_degree,num_features, degree_law)
     
    def change(self):
        pass


        
#make a function that extracts the class of the experiment given the name of the experiment
def get_experiment(name):
    if name == 'Abrupt Covariate Shift':
        return ShiftCovariatesAbrupt
    elif name == 'Community Split':
        return SplitCommunity
    elif name == 'Community Merge':
        return MergeCommunity
    elif name == 'New Community':
        return NewCommunity
    elif name == 'Change of Attribute':

        return ChangeAttribute
    elif name == 'Gradual Covariate Shift':
        return ShiftCovariatesGradual
    elif name == 'No Change':
        return NoChange
    else:
        raise Exception(f'{name} Not Implemented')


