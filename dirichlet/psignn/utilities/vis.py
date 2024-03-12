##################### PACKAGES #################################################
################################################################################
import os 
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
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

def plot_multi_loss(hist_train, hist_val, img_saved_path = None):
    
    plt.figure(figsize=[15,10]) 
    plt.rc('ytick', labelsize=15)
    plt.rc('xtick', labelsize=15)
    plt.subplots_adjust(hspace=0.4)
    fs = 13

    plot1 = plt.subplot2grid((4,4), (0,0), colspan=4)
    plot1.plot(hist_train['loss'], c = 'b', label = "Train")
    plot1.plot(hist_val['loss'], c = 'r', label = "Validation")
    plot1.set_yscale('log')
    # plot1.set_xlabel('Epochs', fontsize = fs)
    plot1.set_ylabel('Train Loss', fontsize = fs)
    plot1.legend()
    plot1.set_title('Training Loss', fontsize = fs)

    plot2 = plt.subplot2grid((4,4), (1,0), colspan=4)
    plot2.plot(hist_train['residual_loss'], c = 'b', label = "Train loss")
    plot2.plot(hist_val['residual_loss'], c = 'r', label = "Val loss")
    plot2.set_yscale('log')
    # plot2.set_xlabel('Epochs', fontsize = fs)
    plot2.set_ylabel('Residual', fontsize = fs)
    plot2.legend()
    plot2.set_title('Residual loss', fontsize = fs)
    
    plot3 = plt.subplot2grid((4,4), (2,0), colspan=4)
    plot3.plot(hist_train['mse_loss'], c = 'b', label = "Train loss")
    plot3.plot(hist_val['mse_loss'], c = 'r', label = "Val loss")
    plot3.set_yscale('log')
    # plot3.set_xlabel('Epochs', fontsize = fs)
    plot3.set_ylabel('MSE', fontsize = fs)
    plot3.legend()
    plot3.set_title('Mean square error (MSE) loss', fontsize = fs)
    
    plot4 = plt.subplot2grid((4,4), (3,0), colspan=4)
    plot4.plot(hist_train['jacobian_loss'], c ='g', label = 'Train loss')
    plot4.set_yscale('log')
    plot4.set_xlabel('Epochs', fontsize = fs)
    plot4.set_ylabel('Jac Frob Norm', fontsize = fs)
    plot4.legend()
    plot4.set_title('Jacobian Frobenius norm loss', fontsize = fs)
    
    plt.savefig(img_saved_path, dpi = 100)

    plt.show()

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
    
    # plt.savefig(img_saved_path, dpi = 200)
    
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
    
    # plt.savefig(img_saved_path, dpi = 200)

    plt.show()

    plt.close()
    
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

    err_seq_max = [np.max(error_list[i]) for i in range(int(0.9*len(sol_list)), len(sol_list))]
    err_seq_min = [np.min(error_list[i]) for i in range(int(0.9*len(sol_list)), len(sol_list))]
    err_max_x = np.max(err_seq_max)
    err_min_x = np.min(err_seq_min)

    xres = np.arange(0, len(res_list))
    xmse = np.arange(0, len(mse_list))

    dot_size = 40
    colormap = "magma"
    
    lxtick = np.arange(0, int(len(sol_list)/2), int(len(sol_list)/2/10))
    for i in tqdm(range(len(lxtick))) : 

        fig = plt.figure(figsize=[10,5]) 
        plt.rc('ytick', labelsize=15)
        plt.rc('xtick', labelsize=15)

        plot = plt.subplot()
        plot.scatter(coordinates[:,0], coordinates[:,1], c = sol_list[lxtick[i]], s = dot_size, vmin = sol_min_x, vmax = sol_max_x, cmap = colormap)
        for j in range(np.shape(edge_index)[1]) :
            ei = edge_index[:,j]
            coord_1 = coordinates[ei[0],:2]
            coord_2 = coordinates[ei[1],:2]
            dx = coord_2[0]-coord_1[0]
            dy = coord_2[1]-coord_1[1]
            plot.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.00001, head_starts_at_zero = True, linewidth = 0.1)
        plot.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plot.set_axis_off()

        plt.savefig(img_saved_path + '_{}.png'.format(lxtick[i]), dpi = 200)
        
        plt.close(fig)

    fig = plt.figure(figsize=[10,5]) 
    plt.rc('ytick', labelsize=15)
    plt.rc('xtick', labelsize=15)

    plot = plt.subplot()
    plot.scatter(coordinates[:,0], coordinates[:,1], c = sol_list[-1], s = dot_size, vmin = sol_min_x, vmax = sol_max_x, cmap = colormap)
    for j in range(np.shape(edge_index)[1]) :
        ei = edge_index[:,j]
        coord_1 = coordinates[ei[0],:2]
        coord_2 = coordinates[ei[1],:2]
        dx = coord_2[0]-coord_1[0]
        dy = coord_2[1]-coord_1[1]
        plot.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.00001, head_starts_at_zero = True, linewidth = 0.1)
    plot.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plot.set_axis_off()

    plt.savefig(img_saved_path + '_{}.png'.format(len(sol_list)), dpi = 200)
    
    plt.close(fig)


    fig = plt.figure(figsize=[10,5]) 
    plt.rc('ytick', labelsize=15)
    plt.rc('xtick', labelsize=15)

    plot = plt.subplot()
    plot.scatter(coordinates[:,0], coordinates[:,1], c = np.asarray(example.sol), s = dot_size, vmin = sol_min_x, vmax = sol_max_x, cmap = colormap)
    for j in range(np.shape(edge_index)[1]) :
        ei = edge_index[:,j]
        coord_1 = coordinates[ei[0],:2]
        coord_2 = coordinates[ei[1],:2]
        dx = coord_2[0]-coord_1[0]
        dy = coord_2[1]-coord_1[1]
        plot.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.00001, head_starts_at_zero = True, linewidth = 0.1)
    plot.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plot.set_axis_off()

    plt.savefig(img_saved_path + 'lu_solution.png', dpi = 200)
    
    plt.close(fig)

def plot_final_gif(example, out, img_saved_path) : 
    
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
    plot4.set_ylim([0.5*MIN_RES_MSE, 2*MAX_RES_MSE])
    plot4.set_xticks(lxtick)
    plot4.set_title("Evolution of Residual and MSE loss across iterations", y = 1.05)

    plt.savefig(img_saved_path, dpi = 300)

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

    err_seq_max = [np.max(error_list[i]) for i in range(int(0.98*len(sol_list)), len(sol_list))]
    err_seq_min = [np.min(error_list[i]) for i in range(int(0.98*len(sol_list)), len(sol_list))]
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

    err_seq_max = [np.max(error_list[i]) for i in range(len(sol_list) - 18, len(sol_list))]
    err_seq_min = [np.min(error_list[i]) for i in range(len(sol_list) - 18, len(sol_list))]
    err_max_x = np.max(err_seq_max)
    err_min_x = np.min(err_seq_min)

    xres = np.arange(0, len(res_list))
    xmse = np.arange(0, len(mse_list))

    MIN_RES_MSE = np.min(mse_list + res_list)
    MAX_RES_MSE = np.max(mse_list + res_list)

    dot_size = 20
    colormap = 'RdBu'

    lxtick = np.arange(0, len(sol_list) - 1, int(len(sol_list)/10))
    lytick = [100, 1, 0.1, 0.01]
    fig = plt.figure(figsize=[7,5], dpi=200) 
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
    plot1.set_title("Data-driven solution")
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
    plot2.set_title("LU solution")
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
    plot3.set_title("Error w.r.t LU")
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
    plot4.set_ylim([0.2*np.min(MIN_RES_MSE), 2*np.max(MAX_RES_MSE)])
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

    plt.savefig("poster.png")

    plt.show() 

    plt.close()

def save_one_sample(sample, solution, img_saved_path):
    
    edge_index = sample.edge_index
    edge_index = edge_index[[1,0],:]
    solution = solution.detach().cpu().numpy()
    coord = sample.pos.numpy()
    min_x = np.min(solution)
    max_x = np.max(solution)

    dot_size = 80

    plt.figure(figsize = (7,7))
    
    plt.subplot(111)
    plt.scatter(coord[:,0], coord[:,1], c = solution, s = dot_size, vmin = min_x, vmax = max_x, cmap = 'plasma')
    for i in range(np.shape(edge_index)[1]) :
        ei = edge_index[:,i]
        coord_1 = coord[ei[0],:2]
        coord_2 = coord[ei[1],:2]
        dx = coord_2[0]-coord_1[0]
        dy = coord_2[1]-coord_1[1]
        plt.arrow(coord_1[0], coord_1[1], dx, dy, width = 0.00001, head_starts_at_zero = True, linewidth = 0.1)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plt.axis('off')    
    plt.savefig(img_saved_path, dpi = 300)
    
    plt.show()

def plot_mse_residual(out, img_saved_path):

    res_list = out["res_dic"]
    mse_list = out["mse_dic"]

    xres = np.arange(0, len(res_list))
    xmse = np.arange(0, len(mse_list))

    MIN_RES_MSE = np.min(mse_list + res_list)
    MAX_RES_MSE = np.max(mse_list + res_list)

    lxtick = np.arange(0, len(res_list) - 1, int(len(res_list)/10))
    lytick = [100, 1, 0.1, 0.01]

    fig = plt.figure(figsize=[18,6], dpi=300) 
    plt.rc('ytick', labelsize=15)
    plt.rc('xtick', labelsize=15)

    plot4 = plt.subplot()
    plot4.plot(xmse, mse_list, 'b-', linewidth = 5, label='MSE')
    plot4.plot(xres, res_list, 'r-', linewidth = 5, label='Residual')
    plot4.set_yscale('log')
    plot4.set_ylabel('')
    plot4.set_xlabel('Iterations')
    plot4.legend()
    plot4.set_ylim([0.2*np.min(MIN_RES_MSE), 2*np.max(MAX_RES_MSE)])
    plot4.set_xticks(lxtick)
    plot4.set_yticks(lytick)

    plt.savefig(img_saved_path, dpi = 200)

    # plt.show()
################################### LOSS PLOT ##################################
################################################################################

def plot_residual_loss(loss_residual):

    plt.rc('ytick', labelsize=15)
    plt.rc('xtick', labelsize=15)

    x = []
    y = []
    for keys, values in loss_residual.items():
        x.append(keys)
        y.append(values.cpu())

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x, y, '-x')
    ax.set_yscale("log")
    ax.set_xlabel("Iterations", fontsize=15)
    ax.set_ylabel("Residual Loss", fontsize=15)

    ax.set_xticks(np.arange(0,len(x),10))
    # ax.set_title("Residual loss w.r.t updates")

def plot_residual_loss_deqdss(loss_residual) :

    x = np.arange(0, len(loss_residual))
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x, loss_residual, '-x')
    ax.set_yscale("log")
    ax.set_xlabel("Iterations", fontsize=15)
    ax.set_ylabel("Residual Loss", fontsize=15)

    ax.set_xticks(np.arange(0,len(x),10))
    # ax.set_title("Residual loss w.r.t updates")

def plot_training_loss(txt_file):
    val_loss = []
    train_loss = []

    with open(txt_file, 'r') as myfile :
        for myline in myfile :
            if 'Validation' in myline :
                l = myline.split()
                val_loss.append(float(l[9]))
            elif 'Training Epoch' in myline :
                l = myline.split()
                train_loss.append(float(l[8]))
    myfile.close()

    x = np.arange(len(val_loss))

    plt.plot(x, val_loss, label = "Validation loss U^K")
    plt.plot(x, train_loss, label = "Training loss U^K")
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.legend()
    plt.title("Training and Validation loss")
    plt.show()

    # return x, val_loss, train_loss

def plot_spectral_radius(txt_file) : 
    radius = []
    with open(txt_file, 'r') as myfile :
        # contents = myfile.readlines()
        # print(contents)
        for myline in myfile :
            l = myline.split()
            radius.append(float(l[3]))
    myfile.close()

    x = np.arange(len(radius))

    plt.plot(x, radius, label = "spectral radius")
    plt.yscale('log')
    plt.xlabel('Validation Epoch')
    plt.legend()
    plt.title("Evolution of spectral radius through val epochs")
    plt.show()