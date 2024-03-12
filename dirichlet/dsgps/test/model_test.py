######################################################################
############################# PACKAGES ###############################
######################################################################
import os 

import numpy as np 

import torch
import torch.nn as nn
from torch_sparse import SparseTensor
import torch.autograd as autograd

from torch_geometric.nn import MessagePassing, MLP
import torch_geometric.nn as geonn
from torch_geometric.utils import remove_self_loops

from utilities import utils 
utils.set_seed()
######################################################################
######################################################################
######################################################################

######################################################################
############################# MAIN MODEL #############################
######################################################################

class ModelDSGPS(nn.Module):

    def __init__(self, config):
        super(ModelDSGPS, self).__init__()

        self.config = config

        #Trainable blocks
        self.laynorm = nn.LayerNorm(self.config["latent_dim"])

        self.phi_to = Phi_to([2*self.config["latent_dim"] + 3, self.config["latent_dim"], self.config["latent_dim"]], nn.ReLU())
        self.phi_from = Phi_from([2*self.config["latent_dim"] + 3, self.config["latent_dim"], self.config["latent_dim"]], nn.ReLU())

        self.z_k = MLPActivation([3*self.config["latent_dim"] + 2, self.config["latent_dim"]], nn.Sigmoid())
        self.r_k = MLPActivation([3*self.config["latent_dim"] + 2, self.config["latent_dim"]], nn.Sigmoid())
        self.correction = MLPActivation([3*self.config["latent_dim"] + 2, self.config["latent_dim"]], nn.Tanh())
        
        self.autoencoder = Autoencoder([1, self.config["latent_dim"], self.config["latent_dim"]], nn.ReLU())

        self.mse_loss = nn.MSELoss()

    def forward(self, batch):

        #Initialisation
        H, U = {}, {}

        cumul_res = {}
        cumul_mse = {}
        cumul_enc = {}
        cumul_autoenc = {}
        cumul_mse_dirichlet = {}
        loss_dic = {}
        total_loss = None

        # a = -1000
        # b = 1000
        # index_interior = torch.where(batch.tags==0)[0]
        # batch.x[index_interior,:] = batch.sol[index_interior,:] + (b - a)*torch.rand_like(batch.sol[index_interior,:]) + a
        # print(a, b)

        index_dirichlet = torch.where(batch.tags==1)[0]

        # Initialize U0
        U['0'] = batch.x

        cumul_res['0'] = self.residual_loss(U['0'], batch)
        cumul_mse['0'] = self.mse_loss(U['0'], batch.sol)
        cumul_mse_dirichlet['0'] = self.mse_loss(U['0'][index_dirichlet,:], batch.sol[index_dirichlet,:])

        # Initialize U_0 and H_0
        H['0'] = self.autoencoder.encoder(U['0'])

        for update in range(self.config["k"]) :

            # INTERIOR MESSAGE PASSING
            mess_to = self.phi_to(H[str(update)], batch.edge_index, batch.edge_attr)
            # mess_to = self.laynorm(mess_to)

            mess_from = self.phi_from(H[str(update)], batch.edge_index, batch.edge_attr)
            # mess_from = self.laynorm(mess_from)

            interior_concat = torch.cat([H[str(update)], mess_to, mess_from, batch.prb_data], dim = 1)
            alpha = self.z_k(interior_concat)
            reset = self.r_k(interior_concat)
            corr = self.correction(torch.cat([reset*H[str(update)], mess_to, mess_from, batch.prb_data], dim = 1))
            interior_next = alpha*corr

            # NEXT LATENT STATE
            H[str(update+1)] = H[str(update)] + interior_next
            H[str(update+1)][index_dirichlet,:] = H["0"][index_dirichlet,:]

            # DECODE TO U(t+1)
            U[str(update+1)] = self.autoencoder.decoder(H[str(update+1)])

            #### COMPUTE LOSSES ####

            # LOSS RESIDUAL
            cumul_res[str(update+1)] = self.residual_loss(U[str(update+1)], batch)
            cumul_mse[str(update+1)] = self.mse_loss(U[str(update+1)], batch.sol)

            # LOSS ENCODER
            for p in self.autoencoder.decoder.parameters() :
                p.requires_grad = False
            cumul_enc[str(update+1)] = self.mse_loss(self.autoencoder(H[str(update+1)], sens = "latent"), H[str(update+1)])
            for p in self.autoencoder.decoder.parameters() :
                p.requires_grad = True

            # LOSS DIRICHLET
            for p in self.autoencoder.encoder.parameters() :
                p.requires_grad = False
            cumul_autoenc[str(update+1)] = self.mse_loss(self.autoencoder(U[str(update+1)], sens = "physics"), U[str(update+1)])
            for p in self.autoencoder.encoder.parameters() :
                p.requires_grad = True

            cumul_mse_dirichlet[str(update+1)] = self.mse_loss(U[str(update+1)][index_dirichlet,:], batch.sol[index_dirichlet,:])

            if total_loss is None :
                total_loss = cumul_res[str(update+1)] * self.config["gamma"]**(self.config["k"] - update - 1) + cumul_enc[str(update+1)] + cumul_autoenc[str(update+1)]
            else :
                total_loss += cumul_res[str(update+1)] * self.config["gamma"]**(self.config["k"] - update - 1) + cumul_enc[str(update+1)] + cumul_autoenc[str(update+1)]
        
        loss_dic["train_loss"] = total_loss
        loss_dic["residual_loss"] = cumul_res
        loss_dic["encoder_loss"] = cumul_enc
        loss_dic["autoencoder_loss"] = cumul_autoenc
        loss_dic["mse_dirichlet_loss"] = cumul_mse_dirichlet
        loss_dic["mse_loss"] = cumul_mse

        return U, loss_dic

    def residual_loss(self, u, batch):
        
        sparse_matrix = SparseTensor(   row = batch.edge_index[0], 
                                        col = batch.edge_index[1], 
                                        value = batch.a_ij.ravel(), 
                                        sparse_sizes=(batch.num_nodes, batch.num_nodes)
                                    )
        
        residual =  sparse_matrix.matmul(u) - batch.y
        
        return torch.mean(residual**2)

######################################################################
######################################################################
######################################################################

######################################################################
############################# DL NETWORKS ############################
######################################################################

def initialize_weights_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class MLP(nn.Module):
    def __init__(self, hidden_channels=None, activation=None):
        super().__init__()

        layers = []
        units = hidden_channels[0]
        for k in range(1, len(hidden_channels)):
            next_units = hidden_channels[k]
            layers.append(nn.Linear(units, next_units))
            if k != len(hidden_channels) - 1 : layers.append(activation)
            units = next_units

        self.mlp = nn.Sequential(*layers).apply(initialize_weights_xavier)

    def forward(self, x):

        return self.mlp(x)

class MLPActivation(nn.Module):
    def __init__(self, hidden_channels=None, activation=None):
        super(MLPActivation, self).__init__()

        layers = []
        units = hidden_channels[0]
        for k in range(1, len(hidden_channels)):
            next_units = hidden_channels[k]
            layers.append(nn.Linear(units, next_units))
            layers.append(activation)
            units = next_units

        self.mlp = nn.Sequential(*layers).apply(initialize_weights_xavier)

    def forward(self, x):

        return self.mlp(x)
  
class Phi_to(MessagePassing):
    def __init__(self, hidden_channels=None, activation=None):
        super(Phi_to, self).__init__(aggr = 'add', flow = 'source_to_target')

        self.mlp = MLP(hidden_channels, activation)

    def forward(self, x, edge_index, edge_attr):
        
        edge_index, edge_attr = remove_self_loops(edge_index=edge_index, edge_attr=edge_attr)

        return self.propagate(edge_index, x = x, edge_attr = edge_attr)

    def message(self, x_i, x_j, edge_attr):

        tmp = torch.cat([x_i, x_j, edge_attr], dim = 1)

        return self.mlp(tmp)

class Phi_from(MessagePassing):
    def __init__(self, hidden_channels=None, activation=None):
        super(Phi_from, self).__init__(aggr = 'add', flow = "target_to_source")

        self.mlp = MLP(hidden_channels, activation)

    def forward(self, x, edge_index, edge_attr):

        edge_index, edge_attr = remove_self_loops(edge_index=edge_index, edge_attr=edge_attr)

        return self.propagate(edge_index, x = x, edge_attr = edge_attr)

    def message(self, x_i, x_j, edge_attr):

        tmp = torch.cat([x_i, x_j, edge_attr], dim = 1)

        return self.mlp(tmp)

class Encoder(nn.Module):
    def __init__(self, hidden_channels=None, activation=None):
        super(Encoder, self).__init__()
        
        self.mlp = MLP(hidden_channels, activation)

    def forward(self, x):

        return self.mlp(x)

class Decoder(nn.Module):
    
    def __init__(self, hidden_channels=None, activation=None):
        super(Decoder, self).__init__()
    
        self.mlp = MLP(hidden_channels, activation)
    
    def forward(self, x):

        return self.mlp(x)

class Autoencoder(nn.Module):
    def __init__(self, hidden_channels=None, activation=None):
        super(Autoencoder, self).__init__()
        
        self.encoder = Encoder(hidden_channels, activation)
        self.decoder = Decoder(list(reversed(hidden_channels)), activation)

    def forward(self, x, sens):
        if sens == "latent" :
            x = self.decoder(x)
            return self.encoder(x)
        elif sens == "physics" :
            x = self.encoder(x)
            return self.decoder(x)
        else :
            print("Specify autoencoder direction")

######################################################################
######################################################################
######################################################################

######################################################################
########################## JACOBIAN LOSS #############################
######################################################################

def jac_loss_estimate(f0, z0, vecs=2, create_graph=True):
    """Estimating tr(J^TJ)=tr(JJ^T) via Hutchinson estimator
    Args:
        f0 (torch.Tensor): Output of the function f (whose J is to be analyzed)
        z0 (torch.Tensor): Input to the function f
        vecs (int, optional): Number of random Gaussian vectors to use. Defaults to 2.
        create_graph (bool, optional): Whether to create backward graph (e.g., to train on this loss). 
                                       Defaults to True.
    Returns:
        torch.Tensor: A 1x1 torch tensor that encodes the (shape-normalized) jacobian loss
    """
    vecs = vecs
    result = 0
    
    for i in range(vecs):
        v = torch.randn(*z0.shape, device = z0.device)
        vJ = torch.autograd.grad(f0, z0, v, retain_graph=True, create_graph=create_graph)[0]
        result += vJ.norm()**2

    return result / vecs / np.prod(z0.shape)

def power_method(f0, z0, n_iters=200):
    """Estimating the spectral radius of J using power method
    Args:
        f0 (torch.Tensor): Output of the function f (whose J is to be analyzed)
        z0 (torch.Tensor): Input to the function f
        n_iters (int, optional): Number of power method iterations. Defaults to 200.
    Returns:
        tuple: (largest eigenvector, largest (abs.) eigenvalue)
    """
    evector = torch.randn_like(z0)
    bsz = 1 
    for i in range(n_iters):
        vTJ = torch.autograd.grad(f0, z0, evector, retain_graph=(i < n_iters-1), create_graph=False)[0]
        evalue = (vTJ * evector).reshape(bsz, -1).sum(1, keepdim=True) / (evector * evector).reshape(bsz, -1).sum(1, keepdim=True)
        evector = (vTJ.reshape(bsz, -1) / vTJ.reshape(bsz, -1).norm(dim=1, keepdim=True)).reshape_as(z0)
    return (evector, torch.abs(evalue))

######################################################################
######################################################################
######################################################################
