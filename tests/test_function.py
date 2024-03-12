### PACKAGES ###
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from tqdm import tqdm
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch_geometric.nn import DataParallel

import time 

from utilities import vis
from utilities import solver 

import model_dss as modtest 

from importlib import reload

reload(modtest)
#################

def make_plot(ax, list_ckpt, loss_type, list_labels):

    for i in range(len(list_ckpt)):
        ckpt = list_ckpt[i][loss_type]
        ax.plot(ckpt, '-', linewidth=1, label = list_labels[i])
    ax.set_xlabel("Epochs")
    ax.set_ylabel(loss_type)
    ax.set_yscale("log")
    ax.legend(fontsize = 6, loc = 'upper right', ncol = 2)

def visualize_losses(list_ckpt, list_labels):

    fig = plt.figure(figsize = [11,10], constrained_layout=True)
    spec = gridspec.GridSpec(ncols=2, nrows=3, figure = fig)

    ax1 = fig.add_subplot(spec[0,0])
    make_plot(ax1, list_ckpt, "loss", list_labels)

    ax2 = fig.add_subplot(spec[0,1])
    make_plot(ax2, list_ckpt, "residual_loss", list_labels)

    ax3 = fig.add_subplot(spec[1,0])
    make_plot(ax3, list_ckpt, "jacobian_loss", list_labels)

    ax4 = fig.add_subplot(spec[1,1])
    make_plot(ax4, list_ckpt, "mse_loss", list_labels)

    ax5 = fig.add_subplot(spec[2,0])
    make_plot(ax5, list_ckpt, "encoder_loss", list_labels)

    ax6 = fig.add_subplot(spec[2,1])
    make_plot(ax6, list_ckpt, "autoencoder_loss", list_labels)

    fig.suptitle("Evolution of training losses through epoch")

    plt.show()

def test_full_dataset(list_ckpt, list_names, dataloader, device):

    data = []

    for i in range(len(list_ckpt)):
        
        print("Evaluation model : ", list_names[i])

        checkpoint = list_ckpt[i]

        config = checkpoint["hyperparameters"]
        print("Default config : ", config)

        # config["solver"] = solver.broyden
        # config["fw_thres"] = 400

        # Load the model
        DEQDSSModel = modtest.ModelDEQDSS(config)
        DEQDSSModel.load_state_dict(checkpoint['state_dict'])
        DEQDSSModel = DataParallel(DEQDSSModel).to(device)

        cumul_test_res_loss, cumul_test_mse_loss = [], []
        cumul_nsteps , cumul_dirichlet = [], []

        # Test the model on the test dataset
        DEQDSSModel.eval()

        with torch.no_grad() :

            for test_data in tqdm(dataloader) :
                
                # Output of the model
                U_sol, loss_dic = DEQDSSModel(test_data)

                cumul_test_res_loss.append(loss_dic["residual_loss"].mean().item())
                cumul_test_mse_loss.append(loss_dic["mse_loss"].mean().item())
                cumul_dirichlet.append(loss_dic["mse_dirichlet"].mean().item())
                cumul_nsteps.append(loss_dic["nsteps"])

        data.append([list_names[i],
                np.mean(cumul_test_res_loss),
                np.mean(cumul_test_mse_loss),
                np.mean(cumul_dirichlet),
                np.ceil(np.mean(cumul_nsteps))
                ])

    print("std Res : ", np.std(cumul_test_res_loss))
    print("std MSE : ", np.std(cumul_test_mse_loss))
    print("std Dirichlet : ", np.std(cumul_dirichlet))
    print("std nsteps : ", np.std(cumul_nsteps))
    headers = ['Name','Residual', 'MSE', 'MSEDirichlet', 'Nstep']
    print(tabulate(data, headers=headers, tablefmt="mixed_grid", floatfmt=".3e"))

def solution_one_sample(checkpoint, data, device):
    
    config = checkpoint["hyperparameters"]
    print(config)

    # Load the model
    DEQDSSModel = modtest.ModelDEQDSS(config)
    DEQDSSModel.load_state_dict(checkpoint['state_dict'])
    DEQDSSModel = DataParallel(DEQDSSModel).to(device)

    # Test the model on the specific sample
    DEQDSSModel.eval()
    with torch.no_grad() :

        # Output of the model
        U_sol, loss_dic = DEQDSSModel([data.to(device)])

        # Compute metrics
        res_loss = loss_dic["residual_loss"].item()
        mse_loss = loss_dic["mse_loss"].mean().item()
        mse_dirichlet = loss_dic["mse_dirichlet"].item()
        nsteps = loss_dic["nsteps"]

    table_data = []
    table_data.append([data.x.size(0), res_loss, mse_loss, mse_dirichlet, nsteps])
    headers = ['Nb nodes','Residual', 'MSE', 'MSEDirichlet', 'Nstep']
    print(tabulate(table_data, headers=headers, tablefmt="mixed_grid", floatfmt=".3e"))

    vis.plot2d_results(U_sol, data, "")

def solution_iterative_process(checkpoint, data, device):
    
    config = checkpoint["hyperparameters"]
    print(config)
    
    # config["solver"] = solver.broyden
    # config["solver"] = solver.forward_iteration

    # Load the model
    DEQDSSModelTest = modtest.ModelDEQDSSIterative(config)
    DEQDSSModelTest.load_state_dict(checkpoint['state_dict'])
    DEQDSSModelTest = DEQDSSModelTest.to(device)

    # DEQDSSModelTest = DataParallel(DEQDSSModelTest).to(device)

    DEQDSSModelTest.eval()
    with torch.no_grad() :

        # Output of the model
        out = DEQDSSModelTest(data.to(device))

    table_data = []
    table_data.append([ data.sol.size(0), 
                        out["res_dic"][-1], 
                        out["mse_dic"][-1], 
                        out["bound_mse_dic"][-1], 
                        out["inter_mse_dic"][-1], 
                        out['nstep']
                        ]
                    )
    headers = ['Nb nodes','Residual', 'MSE', 'MSEDirichlet', 'MSEInterior', 'Nstep']
    print(tabulate(table_data, headers=headers, tablefmt="mixed_grid", floatfmt=".3e"))

    vis.plot_iterative_updates(data.cpu(), out, "")
    # vis.save_images_for_gif(data.cpu(), out, "img/")