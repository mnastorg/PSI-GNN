import os
import argparse
import subprocess

from fenics import *
import numpy as np
from math import *
import meshio
import gmsh

def main(args) :

    path_folder = args.path
    if not os.path.exists(path_folder) :
        os.mkdir(path_folder)

    #########################################################################################################
    ####################################### INITIALISATION OF GMSH ##########################################
    #########################################################################################################

    mesh_name = args.name
    gmsh.initialize()
    gmsh.model.add(mesh_name)

    #########################################################################################################
    ########################################## PARAMETERS ###################################################
    #########################################################################################################

    radius = args.radius

    walls = 'Walls'
    tag_walls = args.tag_walls

    size_boundary = args.size_boundary

    nb_bound_points = args.nb_bound_points

    #########################################################################################################
    ####################################### GENERATE BOUNDARY ###############################################
    #########################################################################################################

    alpha = np.linspace(0, 2*pi, nb_bound_points)

    x_alpha = radius*np.cos(alpha)
    y_alpha = radius*np.sin(alpha)

    L = []

    for i in range(nb_bound_points - 1):
        t = (1-0.75)*np.random.random()+0.75
        x_u = t*x_alpha[i]
        x_v = t*y_alpha[i]
        L.append(i+1)
        gmsh.model.geo.addPoint(x_u, x_v, 0, size_boundary, i+1)
    L.append(1)

    diri = []
    diri.append(gmsh.model.geo.addSpline(L, 1))
    
    # gmsh.model.geo.addCurveLoop([1,2,3,4], 1)
    curveloop = gmsh.model.geo.addCurveLoop([1], 1)

    surface = gmsh.model.geo.addPlaneSurface([1], 1)

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, diri, tag_walls)
    gmsh.model.setPhysicalName(1, tag_walls, walls)

    gmsh.model.addPhysicalGroup(2, [surface], 606)
    gmsh.model.setPhysicalName(2, 606, 'Surface')

    #########################################################################################################
    ############################### GENERATE THE MESH AND SAVING FILES ######################################
    #########################################################################################################

    # SI ON VEUT DES RECTANGLES PLUTOT
    # gmsh.model.mesh.setRecombine(2, 1)
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber('Mesh.MshFileVersion', 2)

    msh_path = os.path.join(path_folder, mesh_name + ".msh")
    gmsh.write(msh_path)
    msh = meshio.read(msh_path)

    #### ONLY IF YOU WANT TO PRINT IT IN PARAVIEW ####
    if args.view == True :
        view_path = os.path.join(path_folder, "view_" + mesh_name + ".xdmf")
        meshio.write(view_path, msh)
    ##################################################

    xml_path = os.path.join(path_folder, mesh_name + ".xml")
    subprocess.check_output('dolfin-convert ' + msh_path + " " + xml_path, shell = True)

    mesh = Mesh(xml_path)
    cd = MeshFunction('size_t', mesh , os.path.join(path_folder, mesh_name + "_physical_region.xml"))
    fd = MeshFunction('size_t', mesh, os.path.join(path_folder, mesh_name + "_facet_region.xml"))
    hdf5_path = os.path.join(path_folder, mesh_name + ".h5")
    hdf5 = HDF5File(mesh.mpi_comm(), hdf5_path, "w")
    hdf5.write(mesh, "/mesh")
    hdf5.write(cd, "/physical")
    hdf5.write(fd, "/facet")

    # subprocess.check_output('meshio convert ' + hdf5_path + " " + os.path.join(path_folder, mesh_name+".xdmf"), shell = True)

    file_path = os.path.join(path_folder, mesh_name + "_info")
    file_tags = open(file_path, "w")
    file_tags.write('############################################################### \n')
    file_tags.write('################## INFO TAGS BOUNDARY CONDITIONS ############## \n')
    file_tags.write('############################################################### \n')
    file_tags.write('Boundaries' + '\t\t' + 'tag_number' + '\n')
    file_tags.write(walls + '\t\t\t' + str(tag_walls) + '\n' )
    file_tags.write('############################################################### \n')
    file_tags.write('################## INFO MESH ELEMENTS ######################### \n')
    file_tags.write('############################################################### \n')
    file_tags.write('Number of nodes : ' + '\t\t' + str(msh.points.shape[0]) + '\n' )
    file_tags.write('Number of triangles : ' + '\t\t' + str(msh.cells_dict['triangle'].shape[0]) + '\n' )
    file_tags.close()

    os.remove(msh_path)
    os.remove(xml_path)
    os.remove(os.path.join(path_folder, mesh_name + "_physical_region.xml"))
    os.remove(os.path.join(path_folder, mesh_name + "_facet_region.xml"))

    gmsh.finalize()

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--path",  type=str, default="2d_mesh", help="Folder to store mesh files")
    parser.add_argument("--name",  type=str, default="mesh_0", help="Name of the mesh")
    parser.add_argument("--radius",  type=float, default=1, help="Set geometry radius i.e diameter")
    parser.add_argument("--nb_bound_points",  type=int, default=10, help="Number of points to build boundary")
    parser.add_argument("--size_boundary",  type=float, default=0.08, help="Mesh size at the boundary")
    parser.add_argument("--tag_walls",  type=int, default=101, help="Dirichlet tag value")
    parser.add_argument('--view', dest='view', action='store_true')
    parser.add_argument('--no_view', dest='view', action='store_false')
    parser.set_defaults(view=False)

    args = parser.parse_args()

    main(args)
