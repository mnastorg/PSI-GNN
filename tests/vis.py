##################### PACKAGES #################################################
################################################################################
import os 

import numpy as np
import torch
from tqdm import tqdm
import math 

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.gridspec as gridspec

import io
import pickle

################################################################################
################################################################################

################################ GRAPH STRUCTURE ###############################
################################################################################

def graph_structure(data) :

    edge_index = np.asarray(data.edge_index)
    edge_index = edge_index[[1,0],:]
    coordinates = np.asarray(data.pos)
    u_sol = np.asarray(data.sol)

    plt.figure()
    for i in range(np.shape(edge_index)[1]) :
        ei = edge_index[:,i]
        coord_1 = coordinates[ei[0],:2]
        coord_2 = coordinates[ei[1],:2]
        dx = coord_2[0]-coord_1[0]
        dy = coord_2[1]-coord_1[1]
        plt.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.0001, head_starts_at_zero = True)

    plt.figure()
    max_x = np.max(u_sol)
    min_x = np.min(u_sol)
    coordinates = np.asarray(data.pos)
    plt.scatter(coordinates[:,0], coordinates[:,1], c=u_sol, zorder=2)
    plt.clim(min_x, max_x)
    plt.colorbar()
    plt.title(r'Solution')

################################### 2D PLOT ####################################
################################################################################

def extract_images_results(example, out, img_saved_path) : 
    
    # Extract information
    sol_list = out["sol_dic"]
    res_list = out["res_dic"]
    mse_list = out["mse_dic"]
    mse_dirichlet_list = out["mse_dirichlet_dic"]

    coordinates = np.asarray(example.pos)
    edge_index = example.edge_index
    edge_index = edge_index[[1,0],:]
    
    error_list = []
    for n in range(len(sol_list)):
        error_list.append((np.asarray(sol_list[n]) - np.asarray(example.x))**2)

    # Bound for colorbar and x / y scale axis
    sol_seq_max = [torch.max(sol_list[i]) for i in range(len(sol_list) - 1, len(sol_list))]
    sol_seq_min = [torch.min(sol_list[i]) for i in range(len(sol_list) - 1, len(sol_list))]
    sol_max_x = np.max(sol_seq_max)
    sol_min_x = np.min(sol_seq_min)

    err_seq_max = [np.max(error_list[i]) for i in range(len(sol_list) - 1, len(sol_list))]
    err_seq_min = [np.min(error_list[i]) for i in range(len(sol_list) - 1, len(sol_list))]
    err_max_x = np.max(err_seq_max)
    err_min_x = np.min(err_seq_min)

    err_seq_max2 = [np.max(error_list[i]) for i in range(len(sol_list) - 1, len(sol_list))]
    err_seq_min2 = [np.min(error_list[i]) for i in range(len(sol_list) - 1, len(sol_list))]
    err_max_x2 = np.max(err_seq_max2)
    err_min_x2 = np.min(err_seq_min2)

    xres = np.arange(1, len(res_list)+1)
    xmse = np.arange(1, len(mse_list)+1)

    MIN_RES_MSE = np.min(mse_list + res_list)
    MAX_RES_MSE = np.max(mse_list + res_list)

    dot_size = 30

    ## Figure 1 : Data-driven solution
    plt.figure(figsize=[5,7]) 
    plt.rc('ytick', labelsize=20)
    plt.rc('xtick', labelsize=20)
    plt.scatter(coordinates[:,0], 
                coordinates[:,1], 
                c = np.asarray(sol_list[-1]), 
                s = dot_size, 
                vmin = sol_min_x, 
                vmax = sol_max_x, 
                cmap = 'RdBu')
    for j in range(np.shape(edge_index)[1]) :
        ei = edge_index[:,j]
        coord_1 = coordinates[ei[0],:2]
        coord_2 = coordinates[ei[1],:2]
        dx = coord_2[0]-coord_1[0]
        dy = coord_2[1]-coord_1[1]
        plt.arrow(coord_1[0], 
                  coord_1[1], 
                  dx, 
                  dy, 
                  width = 0.00001, 
                  head_starts_at_zero = True, 
                  linewidth = 0.1)
    plt.tick_params(axis='both', 
                    which='both', 
                    bottom=False, 
                    top=False, 
                    labelbottom=False, 
                    right=False, 
                    left=False, 
                    labelleft=False)
    plt.axis('off')
    plt.colorbar(orientation = "horizontal")
    plt.savefig(os.path.join(img_saved_path,"data_driven_solution"), 
                dpi = 500, 
                transparent = True)

    ## Figure 2 : Error map
    plt.figure(figsize=[5,7]) 
    plt.rc('ytick', labelsize=20)
    plt.rc('xtick', labelsize=20)
    plt.scatter(coordinates[:,0], 
                coordinates[:,1], 
                c = np.asarray(error_list[-1]), 
                s = dot_size, 
                vmin = err_min_x, 
                vmax = err_max_x, 
                cmap = 'afmhot')
    for j in range(np.shape(edge_index)[1]) :
        ei = edge_index[:,j]
        coord_1 = coordinates[ei[0],:2]
        coord_2 = coordinates[ei[1],:2]
        dx = coord_2[0]-coord_1[0]
        dy = coord_2[1]-coord_1[1]
        plt.arrow(coord_1[0], 
                  coord_1[1], 
                  dx, 
                  dy, 
                  width = 0.00001, 
                  head_starts_at_zero = True, 
                  linewidth = 0.1)
    plt.tick_params(axis='both', 
                    which='both', 
                    bottom=False, 
                    top=False, 
                    labelbottom=False, 
                    right=False, 
                    left=False, 
                    labelleft=False)
    plt.axis('off')
    plt.colorbar(orientation = "horizontal")
    plt.savefig(os.path.join(img_saved_path,"error_map"), 
                dpi = 500, 
                transparent = True)

    ## Figure 3 : Node type
    plt.figure(figsize=[5,7]) 
    tags = 100*np.asarray(example.tags)
    color_mapping = {0: 0, 100: 1}
    colors = [color_mapping[values] for values in tags.flatten()]
    colors_map = ['blue', 'red']
    colors_position = [0, 1]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(colors_position, colors_map)))
    plt.scatter(coordinates[:,0], 
                coordinates[:,1], 
                c = colors, 
                s = dot_size, 
                cmap = cmap)
    for j in range(np.shape(edge_index)[1]) :
        ei = edge_index[:,j]
        coord_1 = coordinates[ei[0],:2]
        coord_2 = coordinates[ei[1],:2]
        dx = coord_2[0]-coord_1[0]
        dy = coord_2[1]-coord_1[1]
        plt.arrow(coord_1[0], 
                  coord_1[1], 
                  dx, 
                  dy, 
                  width = 0.00001, 
                  head_starts_at_zero = True, 
                  linewidth = 0.1)
    plt.tick_params(axis='both', 
                    which='both', 
                    bottom=False, 
                    top=False, 
                    labelbottom=False, 
                    right=False, 
                    left=False, 
                    labelleft=False)
    plt.axis("off")
    cbar = plt.colorbar(orientation = "horizontal", ticks = [0, 1])
    cbar.set_ticklabels(['I', 'D'])
    plt.savefig(os.path.join(img_saved_path,"node_type"), 
                dpi = 500, 
                transparent = True)

    ## Figure 4 : Residual & MSE loss
    plt.figure(figsize=[17,6]) 
    lxtick = [1,5,10,15,20,25,30]
    lytick = [100, 1, 0.1, 0.01, 0.001]
    plt.plot(xmse, mse_list, 'b-', label='MSE')
    plt.plot(xres, res_list, 'r-', label='Residual')
    plt.yscale('log')
    plt.xlabel('Iterations', fontsize = 25)
    plt.legend(fontsize=20, loc = 'lower left')
    plt.ylim([0.2*np.min(MIN_RES_MSE), 1.5*np.max(MAX_RES_MSE)])
    plt.xticks(lxtick, fontsize = 25)
    plt.yticks(lytick, fontsize = 25)
    plt.tight_layout()
    plt.savefig(os.path.join(img_saved_path,"losses"), 
                dpi = 600, 
                transparent = True)
    
    ## Figure 5 : Subfigure Evolution
    for i in [0, 9, 19, 24, 29]:
        plt.figure(figsize=[5,5]) 
        plt.rc('ytick', labelsize=20)
        plt.rc('xtick', labelsize=20)
        plt.scatter(coordinates[:,0], 
                    coordinates[:,1], 
                    c = np.asarray(error_list[i]), 
                    s = dot_size, 
                    vmin = err_min_x2, 
                    vmax = err_max_x2, 
                    cmap = 'afmhot')
        for j in range(np.shape(edge_index)[1]) :
            ei = edge_index[:,j]
            coord_1 = coordinates[ei[0],:2]
            coord_2 = coordinates[ei[1],:2]
            dx = coord_2[0]-coord_1[0]
            dy = coord_2[1]-coord_1[1]
            plt.arrow(coord_1[0], 
                    coord_1[1], 
                    dx, 
                    dy, 
                    width = 0.00001, 
                    head_starts_at_zero = True, 
                    linewidth = 0.1)
        plt.tick_params(axis='both', 
                        which='both', 
                        bottom=False, 
                        top=False, 
                        labelbottom=False, 
                        right=False, 
                        left=False, 
                        labelleft=False)
        plt.axis('off')
        plt.savefig(os.path.join(img_saved_path,"subfigure_{}".format(i)), 
                    dpi = 500, 
                    transparent = True)
        plt.show()

    ## Figure 6 : Evolution MSE Dirichlet
    plt.figure(figsize=[9,6]) 
    lxtick = [1,5,10,15,20,25,30]
    lytick = [100, 1, 0.1, 0.01, 0.001, 0.0001]
    plt.plot(xmse, mse_dirichlet_list, 'k-')
    plt.yscale('log')
    plt.xlabel('Iterations', fontsize = 20)
    plt.ylim([0.2*np.min(MIN_RES_MSE), 1.5*np.max(MAX_RES_MSE)])
    plt.xticks(lxtick, fontsize = 20)
    plt.yticks(lytick, fontsize = 20)
    plt.tight_layout()
    plt.savefig(os.path.join(img_saved_path,"error_boundary"), 
                dpi = 600, 
                transparent = True)
    plt.show()
    plt.close()

################################### LOSS PLOT ##################################
################################################################################

def make_plot(ax, list_ckpt, loss_type, list_labels):

    for i in range(len(list_ckpt)):
        ckpt = list_ckpt[i][loss_type]
        ax.plot(ckpt, '-', linewidth=1, label = list_labels[i])
    ax.set_xlabel("Epochs")
    ax.set_ylabel(loss_type)
    ax.set_yscale("log")
    ax.legend(fontsize = 6, loc = 'upper right', ncol = 2)

def visualize_losses(list_ckpt, list_labels):

    fig = plt.figure(figsize = [11,7], constrained_layout=True)
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure = fig)

    ax1 = fig.add_subplot(spec[0,:])
    make_plot(ax1, list_ckpt, "loss", list_labels)

    ax2 = fig.add_subplot(spec[1,0])
    make_plot(ax2, list_ckpt, "residual_loss", list_labels)

    ax4 = fig.add_subplot(spec[1,1])
    make_plot(ax4, list_ckpt, "mse_loss", list_labels)

    fig.suptitle("Evolution of training losses through epoch")

    plt.show()
