##########################################################################
################################ PACKAGES ################################
##########################################################################

import os
import sys
import argparse
import subprocess
import shutil 
from tqdm import tqdm

import numpy as np
from math import *

import extract_data 
import build_mesh as meshcreator

##########################################################################
##########################################################################
##########################################################################

def generate_data(config) :

    path_mesh = config["path_mesh"]
    path_data = config["path_data"]
    
    n_mesh = config["n_mesh"]
    n_samples = config["n_samples"]

    list_A_sparse_matrix, list_b_matrix, list_sol, list_prb_data, list_tags, list_coordinates, list_distance, list_unit_normal_vector = [], [], [], [], [], [], [], [],

    for n in range(n_mesh):

        radius, hsize = 1.0, 0.075

        # generate meshes
        mesh_dict   = { "path_mesh"     : path_mesh,
                        "name_mesh"     : "mesh_{}".format(n),
                        "hsize"         : hsize,
                        "radius"        : radius,
                        "nb_bound_pts"  : 10,
                        "tag_dirichlet" : 101,
                        "tag_neumann"   : 303,
                        "view"          : False
                        }

        meshcreator.build_mesh(mesh_dict)

        for j in tqdm(range(n_samples)) :
            path_current_mesh = os.path.join(path_mesh, "mesh_{}".format(n))
            A_sparse_matrix, b_matrix, dof_coordinates, sol, prb_data, tags, distance, unit_normal_vector = extract_data.solve_poisson(path_current_mesh, radius)
            list_A_sparse_matrix.append(A_sparse_matrix)
            list_b_matrix.append(b_matrix)
            list_coordinates.append(dof_coordinates)
            list_sol.append(sol)
            list_prb_data.append(prb_data)
            list_tags.append(tags)
            list_distance.append(distance)
            list_unit_normal_vector.append(unit_normal_vector)

    np.save(os.path.join(path_data, "A_sparse_matrix.npy"), list_A_sparse_matrix, allow_pickle = True)
    np.save(os.path.join(path_data, "b_matrix.npy"), list_b_matrix, allow_pickle = True)
    np.save(os.path.join(path_data, "sol.npy"), list_sol, allow_pickle = True)
    np.save(os.path.join(path_data, "prb_data.npy"), list_prb_data, allow_pickle = True)
    np.save(os.path.join(path_data, "tags.npy"), list_tags, allow_pickle = True)
    np.save(os.path.join(path_data, "coordinates.npy"), list_coordinates, allow_pickle = True)
    np.save(os.path.join(path_data, "distance.npy"), list_distance, allow_pickle = True)
    np.save(os.path.join(path_data, "unit_normal_vector.npy"), list_unit_normal_vector, allow_pickle = True)

    seq_nodes = [len(i) for i in list_coordinates]
    mean_number_of_nodes = np.mean(seq_nodes)
    std_number_of_nodes = np.std(seq_nodes)
    min_number_of_nodes = np.min(seq_nodes)
    max_number_of_nodes = np.max(seq_nodes)

    file_path = "data/dataset_info.csv"
    file_tags = open(file_path, "w")
    file_tags.write('############################################################### \n')
    file_tags.write('################## INFO ABOUT THE DATASET  #################### \n')
    file_tags.write('############################################################### \n')
    file_tags.write("Number of different meshes : " + str(n_mesh) + '\n' )
    file_tags.write("Number of samples per meshes : " + str(n_samples) + '\n' )
    file_tags.write("Total number of instances : " + str(n_mesh*n_samples) + '\n')
    file_tags.write('\n')
    file_tags.write("Mean of prb_data : " + str(list(np.around(np.mean(np.vstack(list_prb_data), axis = 0),4))) + '\n')
    file_tags.write("Std of prb_data : " + str(list(np.around(np.std(np.vstack(list_prb_data), axis = 0),4))) + '\n')
    file_tags.write('\n')
    file_tags.write("Mean of distance : " + str(list(np.around(np.mean(np.vstack(list_distance), axis = 0),4))) + '\n')
    file_tags.write("Std of distance : " + str(list(np.around(np.std(np.vstack(list_distance), axis = 0),4))) + '\n')
    file_tags.write('\n')
    file_tags.write("Mean of unit_normal_vector : " + str(list(np.around(np.mean(np.vstack(list_unit_normal_vector), axis = 0),4))) + '\n')
    file_tags.write("Std of unit_normal_vector : " + str(list(np.around(np.std(np.vstack(list_unit_normal_vector), axis = 0),4))) + '\n')
    file_tags.write('\n')
    file_tags.write("Mean number of nodes : " + str(int(mean_number_of_nodes)) + '\n')
    file_tags.write("Std number of nodes : " + str(int(std_number_of_nodes)) + '\n')
    file_tags.write("Min number of nodes : " + str(min_number_of_nodes) + '\n')
    file_tags.write("Max number of nodes : " + str(max_number_of_nodes) + '\n')
    file_tags.write('############################################################### \n')
    file_tags.write('############################################################### \n')
    file_tags.write('############################################################### \n')
    file_tags.close()


if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_mesh",      type=str,   default="mesh/",    help="Folder to store mesh files")
    parser.add_argument("--path_data",      type=str,   default="data/",    help="Folder to store data files")
    parser.add_argument("--n_mesh",         type=int,   default=200,        help="Number of meshes to create")
    parser.add_argument("--n_samples",      type=int,   default=50,         help="Number of samples to create per mesh")

    args = parser.parse_args()

    if os.path.exists(args.path_mesh):
        shutil.rmtree(args.path_mesh)
    os.makedirs(args.path_mesh)

    if os.path.exists(args.path_data):
        shutil.rmtree(args.path_data)
    os.makedirs(args.path_data)

    config = {"path_mesh"   : args.path_mesh,
              "path_data"   : args.path_data,
              "n_mesh"      : args.n_mesh,
              "n_samples"   : args.n_samples
              }
    
    # Generate Data
    generate_data(config)
