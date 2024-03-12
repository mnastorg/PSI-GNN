import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append("..")

import sys 

import subprocess 

import os
import numpy as np
from importlib import reload 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx 
from tqdm import tqdm

import torch
import torch_geometric as geonn
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel


from utilities import solver
from utilities import utils 
from utilities import reader
from utilities import vis

import model_dss as dss
import model_dsgps as dsgps
import model_psignn as psignn

from importlib import reload

# from special_geo import mesh_circle_config as mesh_circle 
from special_geo import build_mesh as msh 
# from special_geo import mesh_square
from special_geo import spec_geo_2 as spc
# from special_geo import mesh_2d


reload(spc)
reload(msh)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

radius = [0.6, 1.0]#, 2.0]#, 4.0, 5.0]#, 7.0]

ckpt_best_dss = torch.load("results_dss/ckpt/best_model.pt")
ckpt_best_dsgps = torch.load("results_dsgps/ckpt/best_model.pt")
ckpt_best_psignn = torch.load("results_psignn/ckpt/best_model.pt")

mean_dss, mean_dsgps, mean_psignn = [], [], []
std_dss, std_dsgps, std_psignn = [], [], []
mean_nnodes, mean_nsteps = [], []
std_nnodes, std_nsteps = [], []
mean_clock_dss, mean_clock_dsgps, mean_clock_psignn = [], [], []
std_clock_dss, std_clock_dsgps, std_clock_psignn = [], [], []

for i in radius:
    
    dssval, dsgpsval, psignnval = [], [], []
    nnodes, psignn_nnsteps = [], []
    dss_clock, dsgps_clock, psignn_clock = [], [], []
    
    for n in range(3):

        np.random.seed(int(n*i + 100))

        torch.cuda.empty_cache()
        
        config = {  "path_mesh"     : "special_geo/mesh_files/",
                    "name_mesh"     : "circle",
                    "radius"        : i,
                    "hsize"         : 0.08,
                    "nb_bound_pts"  : 10,
                    "tag_dirichlet" : 101,
                    "view"          : True
                    }
        
        nb_nodes = msh.build_mesh(config)
        nnodes.append(nb_nodes)
        # print(nnodes)

        data_psignn, data_dss = spc.build_data("special_geo/mesh_files/circle", i)

        # print(data_psignn)
        # print(data_dss)
        
        mse_dss, mse_dsgps, mse_psignn, nsteps_psignn, dss_time, dsgps_time, psignn_time = spc.test_sample([ckpt_best_dss, ckpt_best_dsgps, ckpt_best_psignn], [data_dss, data_psignn], device)
        
        dssval.append(mse_dss)
        dsgpsval.append(mse_dsgps)
        psignnval.append(mse_psignn)
        psignn_nnsteps.append(nsteps_psignn)
        dss_clock.append(dss_time)
        dsgps_clock.append(dsgps_time)
        psignn_clock.append(psignn_time)

        torch.cuda.empty_cache()

    mean_dss.append(np.mean(dssval))
    mean_dsgps.append(np.mean(dsgpsval))
    mean_psignn.append(np.mean(psignnval))
    mean_nnodes.append(np.mean(nnodes))
    mean_nsteps.append(np.mean(psignn_nnsteps))
    mean_clock_dss.append(np.mean(dss_clock))
    mean_clock_dsgps.append(np.mean(dsgps_clock))
    mean_clock_psignn.append(np.mean(psignn_clock))

    std_dss.append(np.std(dssval))
    std_dsgps.append(np.std(dsgpsval))
    std_psignn.append(np.std(psignnval))
    std_nnodes.append(np.std(nnodes))
    std_nsteps.append(np.std(psignn_nnsteps))
    std_clock_dss.append(np.std(dss_clock))
    std_clock_dsgps.append(np.std(dsgps_clock))
    std_clock_psignn.append(np.std(psignn_clock))

path_file = "txtresults"

with open(os.path.join(path_file, "dss_results.csv"), 'w') as f:
    f.write('Mean nb nodes : ')
    f.write(str(mean_nnodes))
    f.write('\n')
    f.write('Mean MSE : ')
    f.write(str(mean_dss))
    f.write('\n')
    f.write('Std MSE : ')
    f.write(str(std_dss))
    f.write('\n')
    f.write('Mean Clock Time :')
    f.write(str(mean_clock_dss))
    f.write('\n')
    f.write('Std Clock Time :')
    f.write(str(std_clock_dss))

with open(os.path.join(path_file, "dssgps_results.csv"), 'w') as f:
    f.write('Mean nb nodes : ')
    f.write(str(mean_nnodes))
    f.write('\n')
    f.write('Mean MSE : ')
    f.write(str(mean_dsgps))
    f.write('\n')
    f.write('Std MSE : ')
    f.write(str(std_dsgps))
    f.write('\n')
    f.write('Mean Clock Time :')
    f.write(str(mean_clock_dsgps))
    f.write('\n')
    f.write('Std Clock Time :')
    f.write(str(std_clock_dsgps))

with open(os.path.join(path_file, "psignn_results.csv"), 'w') as f:
    f.write('Mean nb nodes : ')
    f.write(str(mean_nnodes))
    f.write('\n')
    f.write('Mean MSE : ')
    f.write(str(mean_psignn))
    f.write('\n')
    f.write('Std MSE : ')
    f.write(str(std_psignn))
    f.write('\n')
    f.write('Mean nb steps : ')
    f.write(str(mean_nsteps))
    f.write('\n')
    f.write('Std nb steps : ')
    f.write(str(std_nsteps))
    f.write('\n')
    f.write('Mean Clock Time :')
    f.write(str(mean_clock_psignn))
    f.write('\n')
    f.write('Std Clock Time :')
    f.write(str(std_clock_psignn))
