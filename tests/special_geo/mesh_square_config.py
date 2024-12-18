##########################################################################
################################ PACKAGES ################################
##########################################################################

import os
import subprocess
import shutil
import argparse

from fenics import *
import numpy as np
from math import *
import meshio
import gmsh


def build_mesh(config):

    ####################################################################################
    ##################### PARAMETERS ###################################################
    ####################################################################################
    
    path_mesh = config["path_mesh"]
    name_mesh = config["name_mesh"]

    width = config["width"]
    height = config["height"]

    hsize = config["hsize"]
    print("hsize : ", round(hsize,4))
        
    walls = 'Dirichlet'
    tag_dirichlet = config["tag_dirichlet"]
    
    if not os.path.exists(path_mesh) :
        os.mkdir(path_mesh)

    path_current_mesh = os.path.join(config["path_mesh"], name_mesh)
    if os.path.exists(path_current_mesh):
        shutil.rmtree(path_current_mesh)
    os.makedirs(path_current_mesh)

    ####################################################################################
    ##################### INITIALISE GMSH ##############################################
    ####################################################################################

    gmsh.initialize()
    
    gmsh.model.add(name_mesh)

    #########################################################################################################
    ############################################## MESH CREATION ############################################
    #########################################################################################################

    # RECTANGLE
    # gmsh.model.geo.addPoint(0, 0, 0, hsize, 1)
    # gmsh.model.geo.addPoint(width, 0, 0, hsize, 2)
    # gmsh.model.geo.addPoint(width, height, 0, hsize, 3)
    # gmsh.model.geo.addPoint(0, height, 0, hsize, 4)    
    
    gmsh.model.geo.addPoint(-width/2, -height/2, 0, hsize, 1)
    gmsh.model.geo.addPoint(width/2, -height/2, 0, hsize, 2)
    gmsh.model.geo.addPoint(width/2, height/2, 0, hsize, 3)
    gmsh.model.geo.addPoint(-width/2, height/2, 0, hsize, 4)
    
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)

    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)

    # FULL DOMAIN
    gmsh.model.geo.addPlaneSurface([1], 1)
    
    gmsh.model.geo.synchronize()

    # PHYSICAL GROUPS FOR BOUNDARY CONDITIONS
    gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4], tag_dirichlet)
    gmsh.model.setPhysicalName(1, tag_dirichlet, "dirichlet")

    gmsh.model.addPhysicalGroup(2, [1], 606)
    gmsh.model.setPhysicalName(2, 606, 'Surface')

    # if we want rectangles : 
    # gmsh.model.mesh.setRecombine(2, 1)
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber('Mesh.MshFileVersion', 2)

    msh_file_name = "mesh"

    path_msh_file = os.path.join(path_current_mesh, msh_file_name + ".msh")
    gmsh.write(path_msh_file)
    msh = meshio.read(path_msh_file)

    #### ONLY IF YOU WANT TO PRINT IT IN PARAVIEW ####
    if config["view"] == True :
        path_paraview = os.path.join(path_current_mesh, "paraview_" + msh_file_name + ".xdmf")
        meshio.write(path_paraview, msh)
    ##################################################

    xml_path = os.path.join(path_current_mesh, msh_file_name + ".xml")
    subprocess.check_output('dolfin-convert ' + path_msh_file + " " + xml_path, shell = True)

    mesh = Mesh(xml_path)
    cd = MeshFunction('size_t', mesh , os.path.join(path_current_mesh, msh_file_name + "_physical_region.xml"))
    fd = MeshFunction('size_t', mesh, os.path.join(path_current_mesh, msh_file_name + "_facet_region.xml"))
    hdf5_path = os.path.join(path_current_mesh, msh_file_name + ".h5")
    hdf5 = HDF5File(mesh.mpi_comm(), hdf5_path, "w")
    hdf5.write(mesh, "/mesh")
    hdf5.write(cd, "/physical")
    hdf5.write(fd, "/facet")

    with open(os.path.join(path_current_mesh, "mesh_info.csv"), 'a') as f : 
        f.write('########## MESH INFORMATION ########## \n')
        f.write(walls + '\t\t' + str(tag_dirichlet) + '\n' )
        f.write('Hsize : ' + '\t\t' + str(hsize) + '\n')
        f.write('Number of nodes : ' + '\t\t' + str(msh.points.shape[0]) + '\n' )
        f.write('Number of triangles : ' + '\t\t' + str(msh.cells_dict['triangle'].shape[0]) + '\n' )
        f.close()

    os.remove(path_msh_file)
    os.remove(xml_path)
    os.remove(os.path.join(path_current_mesh, msh_file_name + "_physical_region.xml"))
    os.remove(os.path.join(path_current_mesh, msh_file_name + "_facet_region.xml"))

    gmsh.finalize()

    return msh.points.shape[0]
    
if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_mesh",      type=str,   default="mesh/",    help="Folder to store mesh files")
    parser.add_argument("--name_mesh",      type=str,   default="temp",     help="Name of the mesh")
    parser.add_argument("--hsize",          type=float, default=0.05,       help="Size of the mesh")
    parser.add_argument("--tag_dirichlet",  type=int,   default=101,        help="Tag value for Dirichlet boundary")
    parser.add_argument("--view",           type=bool,  default=False,      help="True if we want XDMF files to display in Paraview")
    args = parser.parse_args()

    config = {  "path_mesh"     : args.path_mesh,
                "name_mesh"     : args.name_mesh,
                "hsize"         : args.hsize,
                "tag_dirichlet" : args.tag_dirichlet,
                "view"          : True
                }
    
    build_mesh(config)