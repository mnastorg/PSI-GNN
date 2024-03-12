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

class ModelPSIGNN(nn.Module):    
    def __init__(self, config):
        super().__init__()

        # HYPERPARAMETER
        self.config = config
        
        self.autoencoder = Autoencoder( hidden_channels = [1, self.config["latent_dim"], self.config["latent_dim"]], 
                                        activation = nn.ReLU()
                                        )
                
        self.config_deq = { "solver"                : self.config["solver"], 
                            "fw_tol"                : self.config["fw_tol"], 
                            "fw_thres"              : self.config["fw_thres"], 
                            "bw_tol"                : self.config["bw_tol"], 
                            "bw_thres"              : self.config["bw_thres"],
                            "path_logs"             : self.config["path_logs"]
                            }

        self.deqdss = DeepEquilibrium(  function = Function(    n_layers = self.config["n_layers"],
                                                                latent_dim = self.config["latent_dim"], 
                                                                edge_features_dim = 3, 
                                                                second_member_dim = 3,
                                                                activation = nn.ReLU()
                                                            ), 
                                        config_deq = self.config_deq
                                        )

        self.mse_loss = nn.MSELoss()
        
    def forward(self, batch): 
        
        loss_dic = {}

        # Compute h_initial
        h_initial = self.autoencoder.encoder(batch.x)

        # Compute the fixed point of f_theta initialized with h_init 
        out = self.deqdss(h_initial, batch)
        h_final = out["result"]
        nsteps = out["nstep"]

        # Decode to find u_final
        u_final = self.autoencoder.decoder(h_final)

        # Compute residual loss
        residual_loss = self.residual_loss(u_final, batch)

        # Compute encoder loss 
        u_detached = u_final.detach()
        h_detached = h_final.detach()

        # Encoder loss
        encoder_loss = self.mse_loss(self.autoencoder.encoder(u_detached), h_detached)

        # Freeze encoder and compute decoder loss
        autoencoder_loss = self.mse_loss(self.autoencoder.decoder(self.autoencoder.encoder(u_detached).detach()), u_detached)

        # Additional losses to print
        mse = self.mse_loss(u_final, batch.sol)

        index_dirichlet = torch.where(batch.tags[:,1]==1)[0]
        mse_dirichlet = self.mse_loss(u_final[index_dirichlet,:], batch.x[index_dirichlet,:])

        loss_dic["residual_loss"] = residual_loss
        # loss_dic["jacobian_loss"] = jacobian_loss
        loss_dic["encoder_loss"] = encoder_loss
        loss_dic["autoencoder_loss"] = autoencoder_loss
        loss_dic["mse_loss"] = mse
        loss_dic["mse_dirichlet_loss"] = mse_dirichlet
        
        return u_final, loss_dic

    def residual_loss(self, u, batch):
        
        sparse_matrix = SparseTensor(   row = batch.edge_index[0], 
                                        col = batch.edge_index[1], 
                                        value = batch.a_ij.ravel(), 
                                        sparse_sizes=(batch.num_nodes, batch.num_nodes)
                                    )
        
        residual =  sparse_matrix.matmul(u) - batch.y
        
        return torch.mean(residual**2)

class ModelPSIGNNIterative(nn.Module):    
    def __init__(self, config):
        super().__init__()

        # HYPERPARAMETER
        self.config = config

        self.autoencoder = Autoencoder( hidden_channels = [1, self.config["latent_dim"], self.config["latent_dim"]], 
                                        activation = nn.ReLU()
                                        )
                
        self.config_deq = { "solver"                : self.config["solver"], 
                            "fw_tol"                : self.config["fw_tol"], 
                            "fw_thres"              : self.config["fw_thres"], 
                            "bw_tol"                : self.config["bw_tol"], 
                            "bw_thres"              : self.config["bw_thres"],
                            "path_logs"             : self.config["path_logs"]
                            }

        self.deqdss = DeepEquilibrium(  function = Function(    n_layers = self.config["n_layers"],
                                                                latent_dim = self.config["latent_dim"], 
                                                                edge_features_dim = 3, 
                                                                second_member_dim = 3,
                                                                activation = nn.ReLU()
                                                            ), 

                                        config_deq = self.config_deq
                                        )

        self.mse_loss = nn.MSELoss()

    def forward(self, batch): 
        
        out_dic = { "sol_dic" : [], 
            "res_dic" : [], 
            "mse_dic" : [], 
            "bound_mse_dic" : [], 
            "inter_mse_dic" : [],
            "nstep" : []
            }

        index_boundary = torch.where(batch.tags[:,1]==1)[0]
        index_interior = torch.where(batch.tags[:,0]==1)[0]

        # a = -1000
        # b = 1000
        # batch.x[index_interior,:] = batch.sol[index_interior,:] + (b - a)*torch.rand_like(batch.sol[index_interior,:]) + a
        # batch.x = batch.sol
        
        # print(batch.x.device) 

        out_dic["sol_dic"].append(batch.x.cpu())
        out_dic["res_dic"].append(self.residual_loss(batch.x, batch).cpu().item())
        out_dic["mse_dic"].append(torch.mean((batch.x - batch.sol)**2).cpu().item())
        out_dic["bound_mse_dic"].append(torch.mean((batch.x[index_boundary,:] - batch.sol[index_boundary,:])**2).cpu().item())
        out_dic["inter_mse_dic"].append(torch.mean((batch.x[index_interior,:] - batch.sol[index_interior,:])**2).cpu().item())
        
        # Compute h_initial
        h_initial = self.autoencoder.encoder(batch.x)

        # Compute the fixed point
        # start_time = time.time()
        out_fw = self.deqdss.forward(h_initial, batch)
        # end_time = time.time()
        # print("Time to compute fixed point : ", round(end_time - start_time, 4))
        
        for i in range(len(out_fw["xest_trace"])) : 
            
            h_star = out_fw["xest_trace"][i]
            # h_intermediate = mask_dirichlet * h_initial + (1 - mask_dirichlet) * h_star
            u_intermediate = self.autoencoder.decoder(h_star)

            out_dic["sol_dic"].append(u_intermediate.cpu())
            out_dic["res_dic"].append(self.residual_loss(u_intermediate, batch).cpu().item())
            out_dic["mse_dic"].append(torch.mean((u_intermediate - batch.sol)**2).cpu().item())
            out_dic["bound_mse_dic"].append(torch.mean((u_intermediate[index_boundary,:] - batch.sol[index_boundary,:])**2).cpu().item())
            out_dic["inter_mse_dic"].append(torch.mean((u_intermediate[index_interior,:] - batch.sol[index_interior,:])**2).cpu().item())
        
        out_dic["nstep"] = out_fw["nstep"]
        
        return out_dic
    
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
############################# DEQ FRAMEWORK ##########################
######################################################################

class DeepEquilibrium(nn.Module):
    def __init__(self, function = None, config_deq = None):
        super().__init__()
        
        self.f = function
        self.config_deq = config_deq
        self.path_logs = self.config_deq["path_logs"]

    def forward(self, H_init, batch):
        
        out_fw = self.config_deq["solver"]( lambda H : self.f(H, H_init, batch), 
                                            H_init, 
                                            threshold = self.config_deq["fw_thres"], 
                                            eps = self.config_deq["fw_tol"]
                                          )
        # H_star = out_fw["result"]

        # with torch.enable_grad():
        #     H_star.requires_grad_()
        #     new_H_star = self.f(H_star, H_init, batch)
        
        # jac_loss = jac_loss_estimate(new_H_star, H_star, vecs = 1)
        
        # compute spectral radius with power iteration
        # _, sradius = power_method(new_H_star, H_star, n_iters = 150)
        # print("\n{}".format(sradius.item()))

        return out_fw

######################################################################
######################################################################
######################################################################

######################################################################
############################# GNN FUNCTION ###########################
######################################################################


class Function(nn.Module):

    def __init__(self, n_layers=None, latent_dim=None, edge_features_dim=None, second_member_dim=None, activation=None):
        super().__init__()
        
        self.n_layers = n_layers

        self.laynorm = nn.LayerNorm(latent_dim)

        self.phi_to_list = nn.ModuleList([Phi_to([2*latent_dim+edge_features_dim, latent_dim, latent_dim], activation) for i in range(self.n_layers)])
        self.phi_from_list = nn.ModuleList([Phi_from([2*latent_dim+edge_features_dim, latent_dim, latent_dim], activation) for i in range(self.n_layers)])

        self.alpha = nn.Sequential(nn.Linear(3*latent_dim + second_member_dim, 1), 
                                   nn.Sigmoid()).apply(initialize_weights_xavier)

        self.update_list = nn.ModuleList([MLP([3*latent_dim+second_member_dim, latent_dim, latent_dim], activation) for i in range(self.n_layers)])
    
        self.phi_neumann = Phi_from([2*latent_dim+edge_features_dim, latent_dim, latent_dim], activation)
        self.update_neumann = MLP([2*latent_dim + second_member_dim + 2, latent_dim, latent_dim], activation) 
        
    def forward(self, h, h_initial, batch):
        
        index_dirichlet = torch.where(batch.tags[:,1] == 1)[0]
        index_neumann = torch.where(batch.tags[:,2] == 1)[0]

        for k in range(self.n_layers):

            mp_to_interior = self.phi_to_list[k](h, batch.edge_index, batch.edge_attr)
            mp_from_interior = self.phi_from_list[k](h, batch.edge_index, batch.edge_attr)
            mp_neumann = self.phi_neumann(h, batch.edge_index, batch.edge_attr)
            
            concat_interior = torch.cat([h, mp_to_interior, mp_from_interior, batch.prb_data], dim = 1)
            alpha = self.alpha(concat_interior)
            update_interior = alpha * self.update_list[k](concat_interior)

            concat_neumann = torch.cat([h, mp_neumann, batch.prb_data, batch.unit_normal_vector], dim = 1)
            update_neumann = self.update_neumann(concat_neumann)

            if k == self.n_layers - 1:
                h_next = h + update_interior
                h_next[index_neumann,:] = update_neumann[index_neumann,:]
                h_next = self.laynorm(h_next)

            else :
                h_next = h + update_interior
                h_next[index_neumann,:] = update_neumann[index_neumann,:]
                
            h_next[index_dirichlet,:] = h_initial[index_dirichlet,:] 

        return h_next
            
        
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
