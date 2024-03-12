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

def plot2d_results(results, example, img_saved_path) : 

    example = example.cpu()
    edge_index = example.edge_index
    edge_index = edge_index[[1,0],:]
    results = results.detach().cpu().numpy()
    coord = example.pos.numpy()
    lu_sol = example.sol.numpy()
    min_x = np.min([results, lu_sol])
    max_x = np.max([results, lu_sol])

    dot_size = 40

    plt.figure(figsize = (20,10))
    
    plt.subplot(1,3,1)
    plt.scatter(coord[:,0], coord[:,1], c = results, s = dot_size, vmin = min_x, vmax = max_x, cmap = 'inferno')
    for i in range(np.shape(edge_index)[1]) :
        ei = edge_index[:,i]
        coord_1 = coord[ei[0],:2]
        coord_2 = coord[ei[1],:2]
        dx = coord_2[0]-coord_1[0]
        dy = coord_2[1]-coord_1[1]
        plt.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.00001, head_starts_at_zero = True, linewidth = 0.1)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plt.title("Data-driven sol")
    plt.colorbar()

    plt.subplot(1,3,2)
    plt.scatter(coord[:,0], coord[:,1], c = lu_sol, s = dot_size, vmin = min_x, vmax = max_x, cmap = 'inferno')
    for i in range(np.shape(edge_index)[1]) :
        ei = edge_index[:,i]
        coord_1 = coord[ei[0],:2]
        coord_2 = coord[ei[1],:2]
        dx = coord_2[0]-coord_1[0]
        dy = coord_2[1]-coord_1[1]
        plt.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.00001, head_starts_at_zero = True, linewidth = 0.1)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plt.title("LU sol")
    plt.colorbar()
    
    plt.subplot(1,3,3)
    plt.scatter(coord[:,0], coord[:,1], c = np.abs(results - example.sol.numpy())**2, s = dot_size, cmap = 'inferno')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plt.title("Squared L2 error")
    plt.colorbar()
    
    plt.savefig(img_saved_path, dpi = 200)
    
    plt.show()
    
def plot_presentation(example, U, loss_residual) : 
    
    coordinates = np.asarray(example.pos)
    seq_max = [torch.max(U[str(i)]) for i in range(len(U))]
    seq_min = [torch.min(U[str(i)]) for i in range(len(U))]
    max_x = np.max(seq_max)
    min_x = np.min(seq_min)
    max_x_sol = np.max(np.asarray(example.x.cpu()))
    min_x_sol = np.min(np.asarray(example.x.cpu()))
    
    MAX = np.max([max_x, max_x_sol]) + 5
    MIN = np.min([min_x, min_x_sol]) - 5
    
    x = []
    y = []
    for keys, values in loss_residual.items():
        x.append(keys)
        y.append(values.cpu())
    
    plt.figure(figsize=[15,10]) 
    plt.rc('ytick', labelsize=15)
    plt.rc('xtick', labelsize=15)

    l = np.linspace(0,len(U)-1,7).astype("int")
    for i in range(int(len(l)/2)+1) :
        plot = plt.subplot2grid((3,4), (0,i))
        plot.scatter(coordinates[:,0], coordinates[:,1], c = np.asarray(U[str(l[i])].cpu()), s = 100, vmin = min_x, vmax = max_x, cmap = 'inferno')
        plot.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plot.set_title("Iter {}".format(l[i]))

    for j in range(int(len(l)/2)+1) : 
        plot = plt.subplot2grid((3,4), (1,j))
        plot.scatter(coordinates[:,0], coordinates[:,1], c = np.asarray(U[str(l[j + int(len(l)/2)])].cpu()), s = 100, vmin = min_x, vmax = max_x, cmap = 'inferno')
        plot.set_title("Iter {}".format(l[j + int(len(l)/2)]))
        plot.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

        # plot.title("Iter {}".format(l[i]))

    plot2 = plt.subplot2grid((3,4),(2,0), colspan=4)
    plot2.plot(x, y, '-o')
    plot2.set_yscale('log')
    plot2.set_xticks(l)
    plot2.set_title("Value of the residual across iterations")

    plt.savefig("test.png", dpi = 200)

def plot_iterative_updates(example, out, img_saved_path) : 
    
    # Extract information
    sol_list = out["sol_dic"]
    res_list = out["res_dic"]
    mse_list = out["mse_dic"]
    bound_list = out["bound_mse_dic"]
    inter_list = out["inter_mse_dic"]
    coordinates = np.asarray(example.pos)
    nb_nodes = len(coordinates)
    edge_index = example.edge_index
    edge_index = edge_index[[1,0],:]
    error_list = []

    for n in range(len(sol_list)):
        error_list.append(np.abs(np.asarray(sol_list[n]) - np.asarray(example.sol))**2)

    # Bound for colorbar and x / y scale axis
    sol_seq_max = [torch.max(sol_list[i]) for i in range(len(sol_list) - 5, len(sol_list))]
    sol_seq_min = [torch.min(sol_list[i]) for i in range(len(sol_list) - 5, len(sol_list))]
    sol_max_x = np.max(sol_seq_max)
    sol_min_x = np.min(sol_seq_min)

    err_seq_max = [np.max(error_list[i]) for i in range(int(0.98*len(sol_list)), len(sol_list))]
    err_seq_min = [np.min(error_list[i]) for i in range(int(0.98*len(sol_list)), len(sol_list))]
    err_max_x = np.max(err_seq_max)
    err_min_x = np.min(err_seq_min)

    xres = np.arange(0, len(res_list))
    xmse = np.arange(0, len(mse_list))

    dot_size = 20
    
    fig = plt.figure(figsize=[15,10]) 
    plt.rc('ytick', labelsize=15)
    plt.rc('xtick', labelsize=15)
    
    # Iterations sur ligne 0 et 1 et toutes les colonnes
    l = np.linspace(0,len(sol_list)-1, 6).astype("int")
    for i in range(int(len(l)/2)+1) :
        plot = plt.subplot2grid((5,4), (0,i))
        plot.scatter(coordinates[:,0], coordinates[:,1], c = error_list[l[i]], s = dot_size, vmin = err_min_x, vmax = err_max_x, cmap = 'inferno')
        plot.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plot.set_title("Iter {}".format(l[i]))
        plot.set_axis_off()

    for j in range(int(len(l)/2)) : 
        plot = plt.subplot2grid((5,4), (1,j))
        plot.scatter(coordinates[:,0], coordinates[:,1], c = error_list[l[j + int(len(l)/2)]], s = dot_size, vmin = err_min_x, vmax = err_max_x, cmap = 'inferno')
        plot.set_title("Iter {}".format(l[j + int(len(l)/2)]))
        plot.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plot.set_axis_off()

    plot7 = plt.subplot2grid((5,4), (1,3))
    p_7 = plot7.scatter(coordinates[:,0], coordinates[:,1], c = error_list[-1], s = dot_size, vmin = err_min_x, vmax = err_max_x, cmap = 'inferno')    
    plot7.set_title("LU Solution")
    plot7.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plot7.set_axis_off()
    plt.colorbar(p_7, ax=plot7)

    # Tableau sur ligne 2 et sur 3 colonnes
    plot3 = plt.subplot2grid((5,4),(2,0), colspan=3)
    plot3.set_axis_off()
    table = plot3.table( 
        cellText = np.array([["{:.2e}".format(res_list[-1]), "{:.2e}".format(mse_list[-1]), "{:.2e}".format(bound_list[-1]), "{:.2e}".format(inter_list[-1])]]),
        rowLabels = ["Values"],  
        colLabels = ["Residual", "MSE full graph", "MSE boundary", "MSE interior"],
        colWidths = [0.19] * 4,
        rowColours = ["lightsteelblue"],  
        colColours = ["lightsteelblue"] * 4, 
        cellLoc = 'center',  
        loc = 'center', 
        ) 
    table.set_fontsize(15)
    table.scale(1, 3) 

    plot4 = plt.subplot2grid((5,4),(2,3))
    p_4 = plot4.scatter(coordinates[:,0], coordinates[:,1], c = np.asarray(example.sol), s = dot_size, vmin = sol_min_x, vmax = sol_max_x, cmap = 'inferno')
    plot4.set_title("Error w.r.t LU solution")
    plot4.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plot4.set_axis_off()
    plt.colorbar(p_4, ax=plot4)

    plot5 = plt.subplot2grid((5,4),(3,0), rowspan = 2, colspan = 4)
    plot5.plot(xres, res_list, 'r-', label='Residual')
    plot5.plot(xmse, mse_list, 'b-', label='MSE')
    plot5.set_yscale('log')
    plot5.set_ylabel('MSE')
    plot5.legend()
    plot5.set_xticks(l)
    
    plt.savefig(img_saved_path, dpi = 200)

    plt.show()

def plot_specific_updates(example, out, img_saved_path) : 
    
    # Extract information
    sol_list = out["sol_dic"]
    res_list = out["res_dic"]
    mse_list = out["mse_dic"]
    bound_list = out["bound_mse_dic"]
    inter_list = out["inter_mse_dic"]
    coordinates = np.asarray(example.pos)
    nb_nodes = len(coordinates)
    edge_index = example.edge_index
    edge_index = edge_index[[1,0],:]
    error_list = []

    for n in range(len(sol_list)):
        error_list.append(np.abs(np.asarray(sol_list[n]) - np.asarray(example.sol))**2)

    # Bound for colorbar and x / y scale axis
    sol_seq_max = [torch.max(sol_list[i]) for i in range(len(sol_list) - 1, len(sol_list))]
    sol_seq_min = [torch.min(sol_list[i]) for i in range(len(sol_list) - 1, len(sol_list))]
    sol_max_x = np.max(sol_seq_max)
    sol_min_x = np.min(sol_seq_min)

    err_seq_max = [np.max(error_list[i]) for i in range(int(0.98*len(sol_list)), len(sol_list))]
    err_seq_min = [np.min(error_list[i]) for i in range(int(0.98*len(sol_list)), len(sol_list))]

    error_max_x = np.max(err_seq_max)
    error_min_x = np.min(err_seq_min)

    xres = np.arange(0, len(res_list))
    xmse = np.arange(0, len(mse_list))

    dot_size = 50
    colormap = "afmhot"
    
    nb_sol = len(error_list)
    print(nb_sol)
    firstpart = list(np.arange(0, int(0.5*len(error_list)), 10))
    secondpart = [int(0.75*len(error_list)), int(len(error_list))]
    # lxtick = firstpart + secondpart
    lxtick = list(np.linspace(0, len(error_list)-1, 4, dtype = 'int'))
    print(lxtick)
    # lxtick.append(len(error_list))
    # print(lxtick)
    size_fig_f1 = [10,4]
    size_figure_house = [8,8]
    for i in tqdm(range(len(lxtick))) : 
        if i == 0:
            fig = plt.figure(figsize=size_fig_f1) 
            plt.rc('ytick', labelsize=15)
            plt.rc('xtick', labelsize=15)
            plot1 = plt.subplot()
            p = plot1.scatter(coordinates[:,0], coordinates[:,1], c = error_list[lxtick[i]], s = dot_size, vmin = err_seq_min[-1], vmax = err_seq_max[-1], cmap = colormap)
            for j in range(np.shape(edge_index)[1]) :
                ei = edge_index[:,j]
                coord_1 = coordinates[ei[0],:2]
                coord_2 = coordinates[ei[1],:2]
                dx = coord_2[0]-coord_1[0]
                dy = coord_2[1]-coord_1[1]
                plot1.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.00001, head_starts_at_zero = True, linewidth = 0.1)
            # plt.colorbar(p, ax=plot1, orientation = "horizontal")
            plot1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            plot1.set_axis_off()
        else : 
            fig = plt.figure(figsize=size_fig_f1) 
            plt.rc('ytick', labelsize=15)
            plt.rc('xtick', labelsize=15)
            plot1 = plt.subplot()
            p = plot1.scatter(coordinates[:,0], coordinates[:,1], c = error_list[lxtick[i]], s = dot_size, vmin = error_min_x, vmax = error_max_x, cmap = colormap)
            for j in range(np.shape(edge_index)[1]) :
                ei = edge_index[:,j]
                coord_1 = coordinates[ei[0],:2]
                coord_2 = coordinates[ei[1],:2]
                dx = coord_2[0]-coord_1[0]
                dy = coord_2[1]-coord_1[1]
                plot1.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.00001, head_starts_at_zero = True, linewidth = 0.1)
            # plt.colorbar(p, ax=plot1, orientation = "horizontal")
            plot1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            plot1.set_axis_off()

        plt.savefig(img_saved_path + 'iter_{}.png'.format(lxtick[i]), dpi = 200)
        plt.close(fig)
    
    tags = np.asarray(example.tags)
    tags[:,1] = 100*tags[:,1]
    tags[:,2] = 200*tags[:,2]
    tags = np.sum(tags, axis= 1)
    # Create a scatter plot with colored markers based on tags
    color_mapping = {1: 'blue', 100: 'red', 200: 'gold'}
    # color_mapping = {1: , 100: 0.5, 200: 1}
    colors = [color_mapping[values] for values in tags.flatten()]
    
    fig = plt.figure(figsize=size_fig_f1) 
    plt.rc('ytick', labelsize=15)
    plt.rc('xtick', labelsize=15)
    plot = plt.subplot()
    plot.scatter(coordinates[:,0], coordinates[:,1], c = colors, s = dot_size, vmin = 1, vmax = 200, cmap = 'inferno')
    for j in range(np.shape(edge_index)[1]) :
        ei = edge_index[:,j]
        coord_1 = coordinates[ei[0],:2]
        coord_2 = coordinates[ei[1],:2]
        dx = coord_2[0]-coord_1[0]
        dy = coord_2[1]-coord_1[1]
        plot.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.00001, head_starts_at_zero = True, linewidth = 0.1)
    plot.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plot.set_axis_off()
    plt.savefig(img_saved_path + 'nodetypes.png', dpi = 200)
    plt.close(fig)


    MIN_RES_MSE = np.min(mse_list + res_list)
    MAX_RES_MSE = np.max(mse_list + res_list)
    lytick = [100, 1, 0.1, 0.01]
    fig = plt.figure(figsize=[20,7], dpi=200) 
    plt.rc('ytick', labelsize=15)
    plt.rc('xtick', labelsize=15)

    plot4 = plt.subplot()
    plot4.plot(xmse, mse_list, 'b-', linewidth = 3, label='MSE')#, markevery = lxtick, markersize = 10)
    plot4.plot(xres, res_list, 'r-', linewidth = 3, label='Residual')#, markevery = lxtick, markersize = 10)
    plot4.set_yscale('log')
    plot4.set_ylabel('')
    plot4.set_xlabel('Iterations')
    plot4.legend()
    plot4.set_ylim([0.2*np.min(MIN_RES_MSE), 1.5*np.max(MAX_RES_MSE)])
    plot4.set_xticks(lxtick)
    plot4.set_yticks(lytick)
    plt.savefig(img_saved_path + 'loss.png', dpi = 200)

    # plt.show()
    plt.close()

def plot_final_gif(example, out) : 
    
    # Extract information
    sol_list = out["sol_dic"]
    res_list = out["res_dic"]
    mse_list = out["mse_dic"]
    bound_list = out["bound_mse_dic"]
    inter_list = out["inter_mse_dic"]
    coordinates = np.asarray(example.pos)
    nb_nodes = len(coordinates)
    edge_index = example.edge_index
    edge_index = edge_index[[1,0],:]
    error_list = []

    MIN_RES_MSE = np.min(mse_list + res_list)
    MAX_RES_MSE = np.max(mse_list + res_list)

    for n in range(len(sol_list)):
        error_list.append(np.abs(np.asarray(sol_list[n]) - np.asarray(example.sol))**2)

    # Bound for colorbar and x / y scale axis
    sol_seq_max = [torch.max(sol_list[i]) for i in range(len(sol_list) - 5, len(sol_list))]
    sol_seq_min = [torch.min(sol_list[i]) for i in range(len(sol_list) - 5, len(sol_list))]
    sol_max_x = np.max(sol_seq_max)
    sol_min_x = np.min(sol_seq_min)

    err_seq_max = [np.max(error_list[i]) for i in range(len(sol_list) - 5, len(sol_list))]
    err_seq_min = [np.min(error_list[i]) for i in range(len(sol_list) - 5, len(sol_list))]
    err_max_x = np.max(err_seq_max)
    err_min_x = np.min(err_seq_min)

    xres = np.arange(0, len(res_list))
    xmse = np.arange(0, len(mse_list))

    dot_size = 20
    # colormap = 'inferno'

    lxtick = np.arange(0, len(sol_list) - 1, int(len(sol_list)/10))

    fig = plt.figure(figsize=[10,7]) 
    plt.rc('ytick', labelsize=12)
    plt.rc('xtick', labelsize=10)
    
    plot1 = plt.subplot2grid((5,4), (0,0), colspan = 2, rowspan = 2)
    p_1 = plot1.scatter(coordinates[:,0], coordinates[:,1], c = np.asarray(sol_list[-1]), s = dot_size, vmin = sol_min_x, vmax = sol_max_x, cmap = "hot")
    for j in range(np.shape(edge_index)[1]) :
        ei = edge_index[:,j]
        coord_1 = coordinates[ei[0],:2]
        coord_2 = coordinates[ei[1],:2]
        dx = coord_2[0]-coord_1[0]
        dy = coord_2[1]-coord_1[1]
        plot1.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.00001, head_starts_at_zero = True, linewidth = 0.1)
    plot1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plot1.set_title("Data-driven solution")
    plot1.set_axis_off()
    plt.colorbar(p_1, ax=plot1)

    plot2 = plt.subplot2grid((5,4),(0,2), colspan = 2, rowspan = 2)
    p_2 = plot2.scatter(coordinates[:,0], coordinates[:,1], c = error_list[-1], s = dot_size, vmin = err_min_x, vmax = err_max_x, cmap = "inferno")
    for j in range(np.shape(edge_index)[1]) :
        ei = edge_index[:,j]
        coord_1 = coordinates[ei[0],:2]
        coord_2 = coordinates[ei[1],:2]
        dx = coord_2[0]-coord_1[0]
        dy = coord_2[1]-coord_1[1]
        plot2.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.00001, head_starts_at_zero = True, linewidth = 0.1)
    plot2.set_title("Error w.r.t LU solution")
    plot2.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plot2.set_axis_off()
    plt.colorbar(p_2, ax=plot2)

    # Tableau sur ligne 2 et sur 3 colonnes
    plot3 = plt.subplot2grid((5,4),(2,0), colspan=4)
    plt.subplots_adjust(hspace=1.2)
    table = plot3.table( 
        cellText = np.array([["{}".format(len(error_list)), "{}".format(nb_nodes), "{:.2e}".format(res_list[-1]), "{:.2e}".format(mse_list[-1]), "{:.2e}".format(bound_list[-1]), "{:.2e}".format(inter_list[-1])]]),
        rowLabels = [""],  
        colLabels = ["$\\bf{Update}$", "Nb of nodes", "Residual", "MSE full graph", "MSE boundary", "MSE interior"],
        colWidths = [0.15] * 6,
        rowColours = ["lightsteelblue"],  
        colColours = ["lightsteelblue"] * 6, 
        cellLoc = 'center',  
        loc = 'upper center', 
        ) 
    plot3.set_axis_off()
    table.set_fontsize(15)
    table.scale(1, 2) 

    plot4 = plt.subplot2grid((5,4),(3,0), rowspan = 2, colspan = 4)
    plot4.plot(xmse, mse_list, 'b-', label='MSE')
    plot4.plot(xres, res_list, 'r-', label='Residual')
    plot4.set_yscale('log')
    plot4.set_ylabel('')
    plot4.legend()
    plot4.set_ylim([0.1*MIN_RES_MSE, 10*MAX_RES_MSE])
    plot4.set_xticks(lxtick)
    plot4.set_title("Evolution of Residual and MSE loss across iterations", y = 1.05)

    plt.savefig("temp_poster.png", dpi = 300)

    plt.show()

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

def save_images_for_gif(example, out, img_saved_path):
    
    # Extract information
    sol_list = out["sol_dic"]
    res_list = out["res_dic"]
    mse_list = out["mse_dic"]
    bound_list = out["bound_mse_dic"]
    inter_list = out["inter_mse_dic"]
    coordinates = np.asarray(example.pos)
    nb_nodes = len(coordinates)
    edge_index = example.edge_index
    edge_index = edge_index[[1,0],:]
    error_list = []

    for n in range(len(sol_list)):
        error_list.append(np.abs(np.asarray(sol_list[n]) - np.asarray(example.sol))**2)

    # Bound for colorbar and x / y scale axis
    sol_seq_max = [torch.max(sol_list[i]) for i in range(len(sol_list) - 5, len(sol_list))]
    sol_seq_min = [torch.min(sol_list[i]) for i in range(len(sol_list) - 5, len(sol_list))]
    sol_max_x = np.max(sol_seq_max)
    sol_min_x = np.min(sol_seq_min)

    err_seq_max = [np.max(error_list[i]) for i in range(int(0.8*len(sol_list)), len(sol_list))]
    err_seq_min = [np.min(error_list[i]) for i in range(int(0.8*len(sol_list)), len(sol_list))]
    err_max_x = np.max(err_seq_max)
    err_min_x = np.min(err_seq_min)

    xres = np.arange(0, len(res_list))
    xmse = np.arange(0, len(mse_list))

    MIN_RES_MSE = np.min(mse_list + res_list)
    MAX_RES_MSE = np.max(mse_list + res_list)

    dot_size = 20

    lxtick = np.arange(0, len(sol_list) - 1, int(len(sol_list)/10))

    fig = plt.figure(figsize=[10,7], dpi = 200)
    plt.rc('ytick', labelsize=12)
    plt.rc('xtick', labelsize=10)

    sub1 = plt.subplot2grid((5,4), (0,0), colspan = 2, rowspan = 2)
    sub1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    sub1.set_title("Data-driven solution")
    sub1.set_axis_off()

    sub2 = plt.subplot2grid((5,4), (0,2), colspan = 2, rowspan = 2)
    sub2.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    sub2.set_title("Data-driven solution")
    sub2.set_axis_off()

    sub3 = plt.subplot2grid((5,4), (2,0), colspan = 4)
    sub3.set_axis_off()
    plt.subplots_adjust(hspace=1.2)

    sub4 = plt.subplot2grid((5,4), (3,0), rowspan = 2, colspan = 4)
    sub4.set_title("Evolution of Residual and MSE loss across iterations", y=1.05)

    for j in range(np.shape(edge_index)[1]) :
        ei = edge_index[:,j]
        coord_1 = coordinates[ei[0],:2]
        coord_2 = coordinates[ei[1],:2]
        dx = coord_2[0]-coord_1[0]
        dy = coord_2[1]-coord_1[1]
        sub1.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.00001, head_starts_at_zero = True, linewidth = 0.1)
        sub2.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.00001, head_starts_at_zero = True, linewidth = 0.1)

    buf = io.BytesIO()
    pickle.dump(fig, buf)

    for i in tqdm(range(len(error_list))) : 
        buf.seek(0)
        new_fig = pickle.load(buf)

        sub1 = new_fig.axes[0]
        sub2 = new_fig.axes[1]
        sub3 = new_fig.axes[2]
        sub4 = new_fig.axes[3]

        p_1 = sub1.scatter(coordinates[:,0], coordinates[:,1], c = np.asarray(sol_list[i]), s = dot_size, vmin = sol_min_x, vmax = sol_max_x, cmap = "hot")
        plt.colorbar(p_1, ax=sub1)

        p_2 = sub2.scatter(coordinates[:,0], coordinates[:,1], c = error_list[i], s = dot_size, vmin = err_min_x, vmax = err_max_x, cmap = "inferno")
        plt.colorbar(p_2, ax=sub2)

        table = sub3.table( 
            cellText = np.array([["{}".format(i), "{}".format(nb_nodes), "{:.2e}".format(res_list[i]), "{:.2e}".format(mse_list[i]), "{:.2e}".format(bound_list[i]), "{:.2e}".format(inter_list[i])]]),
            rowLabels = [""],  
            colLabels = ["$\\bf{Update}$", "Nb of nodes", "Residual", "MSE full graph", "MSE boundary", "MSE interior"],
            colWidths = [0.15] * 6,
            rowColours = ["lightsteelblue"],  
            colColours = ["lightsteelblue"] * 6, 
            cellLoc = 'center',  
            loc = 'upper center', 
            ) 
        table.set_fontsize(15)
        table.scale(1, 2) 

        sub4.plot(xmse[:i], mse_list[:i], 'b-', label='MSE')
        sub4.plot(xres[:i], res_list[:i], 'r-', label='Residual')
        sub4.set_yscale('log')
        sub4.set_ylabel('')
        sub4.legend()
        sub4.set_ylim([0.1*MIN_RES_MSE, 10*MAX_RES_MSE])
        sub4.set_xticks(lxtick)
        
        plt.savefig(os.path.join(img_saved_path, "frame_{}.png".format(i)))

        plt.close(new_fig)

def plot_poster(example, out) : 
    
    # Extract information
    sol_list = out["sol_dic"]
    res_list = out["res_dic"]
    mse_list = out["mse_dic"]
    bound_list = out["bound_mse_dic"]
    inter_list = out["inter_mse_dic"]

    coordinates = np.asarray(example.pos)
    nb_nodes = len(coordinates)
    edge_index = example.edge_index
    edge_index = edge_index[[1,0],:]
    
    error_list = []
    for n in range(len(sol_list)):
        error_list.append((np.asarray(sol_list[n]) - np.asarray(example.sol))**2)

    # Bound for colorbar and x / y scale axis
    sol_seq_max = [torch.max(sol_list[i]) for i in range(len(sol_list) - 5, len(sol_list))]
    sol_seq_min = [torch.min(sol_list[i]) for i in range(len(sol_list) - 5, len(sol_list))]
    sol_max_x = np.max(sol_seq_max)
    sol_min_x = np.min(sol_seq_min)

    err_seq_max = [np.max(error_list[i]) for i in range(len(sol_list) - 10, len(sol_list))]
    err_seq_min = [np.min(error_list[i]) for i in range(len(sol_list) - 10, len(sol_list))]
    err_max_x = np.max(err_seq_max)
    err_min_x = np.min(err_seq_min)

    xres = np.arange(0, len(res_list))
    xmse = np.arange(0, len(mse_list))

    MIN_RES_MSE = np.min(mse_list + res_list)
    MAX_RES_MSE = np.max(mse_list + res_list)

    dot_size = 40
    colormap = 'RdBu'

    lxtick = np.arange(0, len(sol_list) - 1, int(len(sol_list)/10))
    lytick = [100, 1, 0.1, 0.01]
    fig = plt.figure(figsize=[8,7], dpi=400) 
    plt.rc('ytick', labelsize=12)
    plt.rc('xtick', labelsize=10)

    plot1 = plt.subplot2grid((4,6), (0,0), colspan = 2, rowspan = 2)
    p_1 = plot1.scatter(coordinates[:,0], coordinates[:,1], c = np.asarray(sol_list[-1]), s = dot_size, vmin = sol_min_x, vmax = sol_max_x, cmap = colormap)
    for j in range(np.shape(edge_index)[1]) :
        ei = edge_index[:,j]
        coord_1 = coordinates[ei[0],:2]
        coord_2 = coordinates[ei[1],:2]
        dx = coord_2[0]-coord_1[0]
        dy = coord_2[1]-coord_1[1]
        plot1.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.00001, head_starts_at_zero = True, linewidth = 0.1)
    plot1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plot1.set_title("Data-driven")
    plot1.set_axis_off()
    plt.colorbar(p_1, ax=plot1, orientation = "horizontal")

    plot2 = plt.subplot2grid((4,6),(0,2), colspan = 2, rowspan = 2)
    p_2 = plot2.scatter(coordinates[:,0], coordinates[:,1], c = np.asarray(example.sol), s = dot_size, vmin = sol_min_x, vmax = sol_max_x, cmap = colormap)
    for j in range(np.shape(edge_index)[1]) :
        ei = edge_index[:,j]
        coord_1 = coordinates[ei[0],:2]
        coord_2 = coordinates[ei[1],:2]
        dx = coord_2[0]-coord_1[0]
        dy = coord_2[1]-coord_1[1]
        plot2.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.00001, head_starts_at_zero = True, linewidth = 0.1)
    plot2.set_title("LU")
    plot2.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plot2.set_axis_off()
    plt.colorbar(p_2, ax=plot2, orientation = "horizontal")

    plot3 = plt.subplot2grid((4,6),(0,4), colspan = 2, rowspan = 2)
    p_3 = plot3.scatter(coordinates[:,0], coordinates[:,1], c = np.asarray(error_list[-1]), s = dot_size, vmin = err_min_x, vmax = err_max_x, cmap = 'afmhot')
    for j in range(np.shape(edge_index)[1]) :
        ei = edge_index[:,j]
        coord_1 = coordinates[ei[0],:2]
        coord_2 = coordinates[ei[1],:2]
        dx = coord_2[0]-coord_1[0]
        dy = coord_2[1]-coord_1[1]
        plot3.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.00001, head_starts_at_zero = True, linewidth = 0.1)
    plot3.set_title("Squared error")
    plot3.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plot3.set_axis_off()
    plt.colorbar(p_3, ax=plot3, orientation = "horizontal")

    plot4 = plt.subplot2grid((4,4),(2,0), colspan = 4, rowspan = 2)
    plot4.plot(xmse, mse_list, 'b-', label='MSE')
    plot4.plot(xres, res_list, 'r-', label='Residual')
    plot4.set_yscale('log')
    plot4.set_ylabel('')
    plot4.set_xlabel('Iterations')
    plot4.legend()
    plot4.set_ylim([0.2*np.min(MIN_RES_MSE), 1.5*np.max(MAX_RES_MSE)])
    plot4.set_xticks(lxtick)
    plot4.set_yticks(lytick)

    # plot5 = plt.subplot2grid((3,4),(2,0), colspan = 4)
    # plot5.plot(xres, res_list, 'rx-', label='Residual')
    # plot5.set_yscale('log')
    # # plot4.set_ylabel('')
    # plot5.set_xlabel('Iterations')
    # plot5.legend()
    # # plot4.set_ylim([0.1*np.min(res_list), 10*np.max(res_list)])
    # plot5.set_xticks(lxtick)

    plt.savefig("poster.png", dpi = 400)

    plt.show() 

    plt.close()

def plot_paper(example, out, img_saved_path) : 
    
    # Extract information
    sol_list = out["sol_dic"]
    res_list = out["res_dic"]
    mse_list = out["mse_dic"]
    bound_list = out["bound_mse_dic"]
    inter_list = out["inter_mse_dic"]

    coordinates = np.asarray(example.pos)
    nb_nodes = len(coordinates)
    edge_index = example.edge_index
    edge_index = edge_index[[1,0],:]
    
    error_list = []
    for n in range(len(sol_list)):
        error_list.append((np.asarray(sol_list[n]) - np.asarray(example.sol))**2)

    # Bound for colorbar and x / y scale axis
    sol_seq_max = [torch.max(sol_list[i]) for i in range(len(sol_list) - 5, len(sol_list))]
    sol_seq_min = [torch.min(sol_list[i]) for i in range(len(sol_list) - 5, len(sol_list))]
    sol_max_x = np.max(sol_seq_max)
    sol_min_x = np.min(sol_seq_min)

    err_seq_max = [np.max(error_list[i]) for i in range(len(sol_list) - 10, len(sol_list))]
    err_seq_min = [np.min(error_list[i]) for i in range(len(sol_list) - 10, len(sol_list))]
    err_max_x = np.max(err_seq_max)
    err_min_x = np.min(err_seq_min)

    xres = np.arange(0, len(res_list))
    xmse = np.arange(0, len(mse_list))

    MIN_RES_MSE = np.min(mse_list + res_list)
    MAX_RES_MSE = np.max(mse_list + res_list)

    dot_size = 50
    colormap = 'RdBu'

    lxtick = np.arange(0, len(sol_list) - 1, int(len(sol_list)/10))
    lytick = [100, 1, 0.1, 0.01]
    fig = plt.figure(figsize=[11,15]) 
    plt.rc('ytick', labelsize=12)
    plt.rc('xtick', labelsize=10)
    
    # fig.tight_layout(pad = 0.5)
    
    plot1 = plt.subplot2grid((6,4), (0,0), colspan = 2, rowspan = 2)
    p_1 = plot1.scatter(coordinates[:,0], coordinates[:,1], c = np.asarray(sol_list[-1]), s = dot_size, vmin = sol_min_x, vmax = sol_max_x, cmap = colormap)
    for j in range(np.shape(edge_index)[1]) :
        ei = edge_index[:,j]
        coord_1 = coordinates[ei[0],:2]
        coord_2 = coordinates[ei[1],:2]
        dx = coord_2[0]-coord_1[0]
        dy = coord_2[1]-coord_1[1]
        plot1.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.00001, head_starts_at_zero = True, linewidth = 0.1)
    plot1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plot1.set_title("Data-driven solution", fontsize = 12, fontweight="bold")
    plot1.set_axis_off()
    plt.colorbar(p_1, ax=plot1, orientation = "horizontal")

    plot2 = plt.subplot2grid((6,4),(0,2), colspan = 2, rowspan = 2)
    p_2 = plot2.scatter(coordinates[:,0], coordinates[:,1], c = np.asarray(example.sol), s = dot_size, vmin = sol_min_x, vmax = sol_max_x, cmap = colormap)
    for j in range(np.shape(edge_index)[1]) :
        ei = edge_index[:,j]
        coord_1 = coordinates[ei[0],:2]
        coord_2 = coordinates[ei[1],:2]
        dx = coord_2[0]-coord_1[0]
        dy = coord_2[1]-coord_1[1]
        plot2.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.00001, head_starts_at_zero = True, linewidth = 0.1)
    plot2.set_title("LU solution", fontsize = 12, fontweight="bold")
    plot2.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plot2.set_axis_off()
    plt.colorbar(p_2, ax=plot2, orientation = "horizontal")

    plot4 = plt.subplot2grid((6,4),(4,0), colspan = 4, rowspan = 2)
    plot4.plot(xmse, mse_list, 'b-', label='MSE')
    plot4.plot(xres, res_list, 'r-', label='Residual')
    plot4.set_yscale('log')
    plot4.set_ylabel('')
    plot4.set_xlabel('Iterations')
    plot4.legend()
    plot4.set_ylim([0.2*np.min(MIN_RES_MSE), 1.5*np.max(MAX_RES_MSE)])
    plot4.set_title("Evolution of Residual and MSE across iterations", fontsize = 12, fontweight="bold")
    plot4.set_xticks(lxtick)
    plot4.set_yticks(lytick)

    tags = np.asarray(example.tags)
    tags[:,1] = 100*tags[:,1]
    tags[:,2] = 200*tags[:,2]
    tags = np.sum(tags, axis= 1)

    plot3 = plt.subplot2grid((6,4),(2,0), colspan = 2, rowspan = 2)
    p_3 = plot3.scatter(coordinates[:,0], coordinates[:,1], c = tags, s = dot_size, vmin = 1, vmax = 200, cmap = 'inferno')
    for j in range(np.shape(edge_index)[1]) :
        ei = edge_index[:,j]
        coord_1 = coordinates[ei[0],:2]
        coord_2 = coordinates[ei[1],:2]
        dx = coord_2[0]-coord_1[0]
        dy = coord_2[1]-coord_1[1]
        plot3.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.00001, head_starts_at_zero = True, linewidth = 0.1)
    plot3.set_title("Graph : Node types", fontsize = 12, fontweight="bold")
    plot3.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plot3.set_axis_off()
    plt.colorbar(p_3, ax=plot3, orientation = "horizontal")

    plot4 = plt.subplot2grid((6,4),(2,2), colspan = 2, rowspan = 2)
    p_4 = plot4.scatter(coordinates[:,0], coordinates[:,1], c = np.asarray(error_list[-1]), s = dot_size, vmin = err_min_x, vmax = err_max_x, cmap = 'afmhot')
    for j in range(np.shape(edge_index)[1]) :
        ei = edge_index[:,j]
        coord_1 = coordinates[ei[0],:2]
        coord_2 = coordinates[ei[1],:2]
        dx = coord_2[0]-coord_1[0]
        dy = coord_2[1]-coord_1[1]
        plot4.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.00001, head_starts_at_zero = True, linewidth = 0.1)
    plot4.set_title("Error map", fontsize = 12, fontweight="bold")
    plot4.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plot4.set_axis_off()
    plt.colorbar(p_4, ax=plot4, orientation = "horizontal")

    # plt.subplot_tool()
    plt.subplots_adjust(top=0.969,
                        bottom=0.052,
                        left=0.1,
                        right=0.9,
                        hspace=0.6,
                        wspace=0.6
                        )    
    plt.savefig(img_saved_path, dpi = 500)

    plt.show() 

    plt.close()

def plot_paper_2(example, out, img_saved_path) : 
    
    # Extract information
    sol_list = out["sol_dic"]
    res_list = out["res_dic"]
    mse_list = out["mse_dic"]

    coordinates = np.asarray(example.pos)
    edge_index = example.edge_index
    edge_index = edge_index[[1,0],:]
    
    error_list = []
    for n in range(len(sol_list)):
        error_list.append((np.asarray(sol_list[n]) - np.asarray(example.sol))**2)

    # Bound for colorbar and x / y scale axis
    sol_seq_max = [torch.max(sol_list[i]) for i in range(len(sol_list) - 1, len(sol_list))]
    sol_seq_min = [torch.min(sol_list[i]) for i in range(len(sol_list) - 1, len(sol_list))]
    sol_max_x = np.max(sol_seq_max)
    sol_min_x = np.min(sol_seq_min)

    err_seq_max = [np.max(error_list[i]) for i in range(len(sol_list) - 1, len(sol_list))]
    err_seq_min = [np.min(error_list[i]) for i in range(len(sol_list) - 1, len(sol_list))]
    err_max_x = np.max(err_seq_max)
    err_min_x = np.min(err_seq_min)

    xres = np.arange(0, len(res_list))
    xmse = np.arange(0, len(mse_list))

    MIN_RES_MSE = np.min(mse_list + res_list)
    MAX_RES_MSE = np.max(mse_list + res_list)

    dot_size = 20
    colormap = 'RdBu'

    lxtick = np.arange(0, len(sol_list), int(len(sol_list)/10))
    lytick = [100, 1, 0.1, 0.01, 0.001]

    fig = plt.figure(figsize=[10,9]) 

    plt.rc('ytick', labelsize=20)
    plt.rc('xtick', labelsize=20)

    plot1 = plt.subplot2grid((4,6), (0,0), colspan = 2, rowspan = 2)
    p_1 = plot1.scatter(coordinates[:,0], coordinates[:,1], c = np.asarray(sol_list[-1]), s = dot_size, vmin = sol_min_x, vmax = sol_max_x, cmap = colormap)
    # for j in range(np.shape(edge_index)[1]) :
    #     ei = edge_index[:,j]
    #     coord_1 = coordinates[ei[0],:2]
    #     coord_2 = coordinates[ei[1],:2]
    #     dx = coord_2[0]-coord_1[0]
    #     dy = coord_2[1]-coord_1[1]
    #     plot1.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.00001, head_starts_at_zero = True, linewidth = 0.1)
    plot1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plot1.set_title("Data-driven solution", fontsize = 15)
    plot1.set_axis_off()
    plt.colorbar(p_1, ax=plot1, orientation = "horizontal")

    plot4 = plt.subplot2grid((4,6),(2,0), colspan = 6, rowspan = 2)
    plot4.plot(xmse, mse_list, 'b-', label='MSE')
    plot4.plot(xres, res_list, 'r-', label='Residual')
    plot4.set_yscale('log')
    plot4.set_ylabel('')
    plot4.set_xlabel('Iterations', fontsize = 14)
    plot4.legend(fontsize=17)
    plot4.set_ylim([0.2*np.min(MIN_RES_MSE), 1.5*np.max(MAX_RES_MSE)])
    plot4.set_title("Evolution of Residual and MSE across iterations", fontsize = 15)
    plot4.set_xticks(lxtick)
    plot4.set_yticks(lytick)

    tags = 100*np.asarray(example.tags)

    # Create a scatter plot with colored markers based on tags
    color_mapping = {0: 0, 100: 1}
    colors = [color_mapping[values] for values in tags.flatten()]

    colors_map = ['blue', 'red']
    colors_position = [0, 1]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(colors_position, colors_map)))

    plot3 = plt.subplot2grid((4,6),(0,2), colspan = 2, rowspan = 2)
    p_3 = plot3.scatter(coordinates[:,0], coordinates[:,1], c = colors, s = dot_size, cmap = cmap)
    # for j in range(np.shape(edge_index)[1]) :
    #     ei = edge_index[:,j]
    #     coord_1 = coordinates[ei[0],:2]
    #     coord_2 = coordinates[ei[1],:2]
    #     dx = coord_2[0]-coord_1[0]
    #     dy = coord_2[1]-coord_1[1]
    #     plot3.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.00001, head_starts_at_zero = True, linewidth = 0.1)
    plot3.set_title("Node types", fontsize = 15)
    plot3.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plot3.set_axis_off()
    cbar = plt.colorbar(p_3, orientation = "horizontal", ticks = [0, 1])
    # cbar.set_ticks([1,100,200])
    cbar.set_ticklabels(['Int.', 'Dir.'])

    plot4 = plt.subplot2grid((4,6),(0,4), colspan = 2, rowspan = 2)
    p_4 = plot4.scatter(coordinates[:,0], coordinates[:,1], c = np.asarray(error_list[-1]), s = dot_size, vmin = err_min_x, vmax = err_max_x, cmap = 'afmhot')
    # for j in range(np.shape(edge_index)[1]) :
    #     ei = edge_index[:,j]
    #     coord_1 = coordinates[ei[0],:2]
    #     coord_2 = coordinates[ei[1],:2]
    #     dx = coord_2[0]-coord_1[0]
    #     dy = coord_2[1]-coord_1[1]
    #     plot4.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.00001, head_starts_at_zero = True, linewidth = 0.1)
    plot4.set_title("Error map", fontsize = 15)
    plot4.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plot4.set_axis_off()
    plt.colorbar(p_4, ax=plot4, orientation = "horizontal")

    # plt.subplot_tool()
    plt.subplots_adjust(top=0.954,
                        bottom=0.072,
                        left=0.09,
                        right=0.9,
                        hspace=0.6,
                        wspace=0.6
                        )
    
    fig.tight_layout()    
    
    plt.savefig(img_saved_path, dpi = 800, transparent = True)

    plt.show() 

    plt.close()

def extract_images_results(example, out, img_saved_path) : 
    
    # Extract information
    sol_list = out["sol_dic"]
    res_list = out["res_dic"]
    mse_list = out["mse_dic"]
    mse_dirichlet_list = out["bound_mse_dic"]

    coordinates = np.asarray(example.pos)
    edge_index = example.edge_index
    edge_index = edge_index[[1,0],:]
    
    error_list = []
    for n in range(len(sol_list)):
        error_list.append((np.asarray(sol_list[n]) - np.asarray(example.sol))**2)

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
    plt.figure(figsize=[15,6]) 
    lxtick = np.linspace(1,len(xmse),10, dtype = 'int')
    lytick = [100, 1, 0.1, 0.01, 0.001]
    plt.plot(xmse, mse_list, 'b-', label='MSE')
    plt.plot(xres, res_list, 'r-', label='Residual')
    plt.yscale('log')
    plt.xlabel('Iterations', fontsize = 25)
    plt.legend(fontsize=20, loc = 'upper right')
    plt.ylim([0.2*np.min(MIN_RES_MSE), 1.5*np.max(MAX_RES_MSE)])
    plt.xticks(lxtick, fontsize = 25)
    plt.yticks(lytick, fontsize = 25)
    plt.tight_layout()
    plt.savefig(os.path.join(img_saved_path,"losses"), 
                dpi = 400, 
                transparent = True)
    
    # ## Figure 5 : Subfigure Evolution
    # for i in [0, 9, 19, 24, 29]:
    #     plt.figure(figsize=[5,5]) 
    #     plt.rc('ytick', labelsize=20)
    #     plt.rc('xtick', labelsize=20)
    #     plt.scatter(coordinates[:,0], 
    #                 coordinates[:,1], 
    #                 c = np.asarray(error_list[i]), 
    #                 s = dot_size, 
    #                 vmin = err_min_x2, 
    #                 vmax = err_max_x2, 
    #                 cmap = 'afmhot')
    #     for j in range(np.shape(edge_index)[1]) :
    #         ei = edge_index[:,j]
    #         coord_1 = coordinates[ei[0],:2]
    #         coord_2 = coordinates[ei[1],:2]
    #         dx = coord_2[0]-coord_1[0]
    #         dy = coord_2[1]-coord_1[1]
    #         plt.arrow(coord_1[0], 
    #                 coord_1[1], 
    #                 dx, 
    #                 dy, 
    #                 width = 0.00001, 
    #                 head_starts_at_zero = True, 
    #                 linewidth = 0.1)
    #     plt.tick_params(axis='both', 
    #                     which='both', 
    #                     bottom=False, 
    #                     top=False, 
    #                     labelbottom=False, 
    #                     right=False, 
    #                     left=False, 
    #                     labelleft=False)
    #     plt.axis('off')
    #     plt.savefig(os.path.join(img_saved_path,"subfigure_{}".format(i)), 
    #                 dpi = 500, 
    #                 transparent = True)
    #     plt.show()

    ## Figure 6 : Evolution MSE Dirichlet
    plt.figure(figsize=[9,6]) 
    lxtick = [1,5,10,15,20,25,30]
    lytick = [100, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
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

def plot_multi_residual(res_list, label_list, img_saved_path):

    plt.figure(figsize=[15,6])
    for i in range(len(res_list)):
        plt.plot(res_list[i]["residual_loss"], '-', label=label_list[i])
    plt.yscale('log')
    plt.xlabel('Epochs', fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.xticks(fontsize = 25)
    plt.tight_layout()    
    plt.tight_layout()
    plt.legend(fontsize = 20)
    plt.savefig(os.path.join(img_saved_path,"multi_losses"), 
                dpi = 400, 
                transparent = True)

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

    ax1 = fig.add_subplot(spec[0,:])
    make_plot(ax1, list_ckpt, "loss", list_labels)

    ax2 = fig.add_subplot(spec[1,0])
    make_plot(ax2, list_ckpt, "residual_loss", list_labels)

    ax4 = fig.add_subplot(spec[1,1])
    make_plot(ax4, list_ckpt, "mse_loss", list_labels)

    ax5 = fig.add_subplot(spec[2,0])
    make_plot(ax5, list_ckpt, "encoder_loss", list_labels)

    ax6 = fig.add_subplot(spec[2,1])
    make_plot(ax6, list_ckpt, "autoencoder_loss", list_labels)

    fig.suptitle("Evolution of training losses through epoch")

    plt.show()

def losses_on_same_plot(ckpt, img_saved_path):
    
    train_loss = ckpt["loss"]
    residual_loss = ckpt["residual_loss"]
    mse_loss = ckpt["mse_loss"]

    plt.figure(figsize=[15,6])
    plt.plot(train_loss, linewidth = 2, label = "Train loss")
    plt.plot(residual_loss, linewidth = 2, c = 'r' , label = "Residual")
    # plt.plot(mse_loss, label = "MSE")
    plt.yscale('log')
    plt.xlabel('Epochs', fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.xticks(fontsize = 25)
    plt.tight_layout()
    plt.legend(fontsize = 20)
    plt.savefig(os.path.join(img_saved_path, "training_losses"), 
                dpi = 600, 
                transparent = True)

    plt.show()