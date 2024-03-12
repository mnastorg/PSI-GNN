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

##########################################################################
##########################################################################
##########################################################################

def build_mesh(config) :

    ####################################################################################
    ##################### PARAMETERS ###################################################
    ####################################################################################
    
    path_mesh = config["path_mesh"]
    name_mesh = config["name_mesh"]

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

    # CAR BOTTOM PART 
    gmsh.model.geo.addPoint(-1.0, -0.55, 0, hsize, 1)
    gmsh.model.geo.addPoint(-0.8, -0.55, 0, hsize, 2)
    gmsh.model.geo.addPoint(-0.8, -0.1, 0, hsize, 3)
    gmsh.model.geo.addPoint(-0.55, -0.138, 0, hsize, 4)
    gmsh.model.geo.addPoint(-0.55, -0.3, 0, hsize, 5)
    gmsh.model.geo.addPoint(-0.65, -0.3, 0, hsize, 6)
    gmsh.model.geo.addPoint(-0.65, -0.45, 0, hsize, 7)
    gmsh.model.geo.addPoint(-0.35, -0.45, 0, hsize, 8)
    gmsh.model.geo.addPoint(-0.35, -0.3, 0, hsize, 9)
    gmsh.model.geo.addPoint(-0.45, -0.3, 0, hsize, 10)
    gmsh.model.geo.addPoint(-0.45, -0.154, 0, hsize, 11)
        
    gmsh.model.geo.addPoint(-0.15, -0.2, 0, hsize, 12)

    gmsh.model.geo.addPoint(-0.1, -0.25, 0, hsize, 13)
    gmsh.model.geo.addPoint(0.0, -0.32, 0, hsize, 14)
    gmsh.model.geo.addPoint(0.1, -0.32, 0, hsize, 15)
    gmsh.model.geo.addPoint(0.2, -0.3, 0, hsize, 16)
    gmsh.model.geo.addPoint(0.25, -0.25, 0, hsize, 17)
    gmsh.model.geo.addPoint(0.3, -0.2, 0, hsize, 18)

    gmsh.model.geo.addPoint(0.5, -0.1, 0, hsize, 19)
    
    gmsh.model.geo.addPoint(0.6, -0.1, 0, hsize, 20)
    gmsh.model.geo.addPoint(0.6, -0.3, 0, hsize, 21)
    gmsh.model.geo.addPoint(0.5, -0.3, 0, hsize, 22)
    gmsh.model.geo.addPoint(0.5, -0.45, 0, hsize, 23)
    gmsh.model.geo.addPoint(0.8, -0.45, 0, hsize, 24)
    gmsh.model.geo.addPoint(0.8, -0.3, 0, hsize, 25)
    gmsh.model.geo.addPoint(0.7, -0.3, 0, hsize, 26)
    gmsh.model.geo.addPoint(0.7, -0.1, 0, hsize, 27)
    gmsh.model.geo.addPoint(0.85, -0.1, 0, hsize, 28)
    gmsh.model.geo.addPoint(0.85, -0.25, 0, hsize, 29)
    gmsh.model.geo.addPoint(1.0, -0.25, 0, hsize, 30)


    # CAR TOP PART 
    gmsh.model.geo.addPoint(-1.0, 0.55, 0, hsize, 60)
    gmsh.model.geo.addPoint(-0.8, 0.55, 0, hsize, 59)
    gmsh.model.geo.addPoint(-0.8, 0.1, 0, hsize, 58)
    gmsh.model.geo.addPoint(-0.55, 0.138, 0, hsize, 57)
    gmsh.model.geo.addPoint(-0.55, 0.3, 0, hsize, 56)
    gmsh.model.geo.addPoint(-0.65, 0.3, 0, hsize, 55)
    gmsh.model.geo.addPoint(-0.65, 0.45, 0, hsize, 54)
    gmsh.model.geo.addPoint(-0.35, 0.45, 0, hsize, 53)
    gmsh.model.geo.addPoint(-0.35, 0.3, 0, hsize, 52)
    gmsh.model.geo.addPoint(-0.45, 0.3, 0, hsize, 51)
    gmsh.model.geo.addPoint(-0.45, 0.154, 0, hsize, 50)
        
    gmsh.model.geo.addPoint(-0.15, 0.2, 0, hsize, 49)

    gmsh.model.geo.addPoint(-0.1, 0.25, 0, hsize, 48)
    gmsh.model.geo.addPoint(0.0, 0.32, 0, hsize, 47)
    gmsh.model.geo.addPoint(0.1, 0.32, 0, hsize, 46)
    gmsh.model.geo.addPoint(0.2, 0.3, 0, hsize, 45)
    gmsh.model.geo.addPoint(0.25, 0.25, 0, hsize, 44)
    gmsh.model.geo.addPoint(0.3, 0.2, 0, hsize, 43)

    gmsh.model.geo.addPoint(0.5, 0.1, 0, hsize, 42)
    
    gmsh.model.geo.addPoint(0.6, 0.1, 0, hsize, 41)
    gmsh.model.geo.addPoint(0.6, 0.3, 0, hsize, 40)
    gmsh.model.geo.addPoint(0.5, 0.3, 0, hsize, 39)
    gmsh.model.geo.addPoint(0.5, 0.45, 0, hsize, 38)
    gmsh.model.geo.addPoint(0.8, 0.45, 0, hsize, 37)
    gmsh.model.geo.addPoint(0.8, 0.3, 0, hsize, 36)
    gmsh.model.geo.addPoint(0.7, 0.3, 0, hsize, 35)
    gmsh.model.geo.addPoint(0.7, 0.1, 0, hsize, 34)
    gmsh.model.geo.addPoint(0.85, 0.1, 0, hsize, 33)
    gmsh.model.geo.addPoint(0.85, 0.25, 0, hsize, 32)
    gmsh.model.geo.addPoint(1.0, 0.25, 0, hsize, 31)

    # LINES BOTTOM 
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 5, 4)
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 7, 6)
    gmsh.model.geo.addLine(7, 8, 7)
    gmsh.model.geo.addLine(8, 9, 8)
    gmsh.model.geo.addLine(9, 10, 9)
    gmsh.model.geo.addLine(10, 11, 10)
    gmsh.model.geo.addLine(11, 12, 11)

    gmsh.model.geo.addSpline([12,13,14,15,16,17,18,19], 12)

    gmsh.model.geo.addLine(19, 20, 13)
    gmsh.model.geo.addLine(20, 21, 14)
    gmsh.model.geo.addLine(21, 22, 15)
    gmsh.model.geo.addLine(22, 23, 16)
    gmsh.model.geo.addLine(23, 24, 17)
    gmsh.model.geo.addLine(24, 25, 18)
    gmsh.model.geo.addLine(25, 26, 19)
    gmsh.model.geo.addLine(26, 27, 20)
    gmsh.model.geo.addLine(27, 28, 21)
    gmsh.model.geo.addLine(28, 29, 22)
    gmsh.model.geo.addLine(29, 30, 23)

    gmsh.model.geo.addLine(30, 31, 24)

    # LINES TOP 
    gmsh.model.geo.addLine(31, 32, 25)
    gmsh.model.geo.addLine(32, 33, 26)
    gmsh.model.geo.addLine(33, 34, 27)
    gmsh.model.geo.addLine(34, 35, 28)
    gmsh.model.geo.addLine(35, 36, 29)
    gmsh.model.geo.addLine(36, 37, 30)
    gmsh.model.geo.addLine(37, 38, 31)
    gmsh.model.geo.addLine(38, 39, 32)
    gmsh.model.geo.addLine(39, 40, 33)
    gmsh.model.geo.addLine(40, 41, 34)
    gmsh.model.geo.addLine(41, 42, 35)

    gmsh.model.geo.addSpline([42,43,44,45,46,47,48,49], 36)

    gmsh.model.geo.addLine(49, 50, 37)
    gmsh.model.geo.addLine(50, 51, 38)
    gmsh.model.geo.addLine(51, 52, 39)
    gmsh.model.geo.addLine(52, 53, 40)
    gmsh.model.geo.addLine(53, 54, 41)
    gmsh.model.geo.addLine(54, 55, 42)
    gmsh.model.geo.addLine(55, 56, 43)
    gmsh.model.geo.addLine(56, 57, 44)
    gmsh.model.geo.addLine(57, 58, 45)
    gmsh.model.geo.addLine(58, 59, 46)
    gmsh.model.geo.addLine(59, 60, 47)

    gmsh.model.geo.addLine(60, 1, 48)
    
    L = np.arange(1,49)
    gmsh.model.geo.addCurveLoop(L, 1)

    ## COCKPIT 
    c_x = -0.1 
    c_y = 0.0
    R = 0.1
    gmsh.model.geo.addPoint(c_x, c_y, 0, hsize, 61)
    gmsh.model.geo.addPoint(c_x, c_y + R, 0, hsize, 62)
    gmsh.model.geo.addPoint(c_x, c_y - R, 0, hsize, 63)
    gmsh.model.geo.addCircleArc(63, 61, 62, 49)
    gmsh.model.geo.addLine(62, 63, 50)

    gmsh.model.geo.addCurveLoop([49, 50], 2)

    ## BOTTOM STRIE 1
    x_A = -0.95
    y_A = -0.35
    tx = 0.1
    ty = 0.05
    gmsh.model.geo.addPoint(x_A, y_A, 0, hsize, 64)
    gmsh.model.geo.addPoint(x_A + tx, y_A - ty, 0, hsize, 65)
    gmsh.model.geo.addPoint(x_A + tx, y_A, 0, hsize, 66)
    gmsh.model.geo.addPoint(x_A, y_A + ty, 0, hsize, 67)
    gmsh.model.geo.addLine(64, 65, 51)
    gmsh.model.geo.addLine(65, 66, 52)
    gmsh.model.geo.addLine(66, 67, 53)
    gmsh.model.geo.addLine(67, 64, 54)

    gmsh.model.geo.addCurveLoop([51, 52, 53, 54], 3)

    ## BOTTOM STRIE 2
    x_A = -0.95
    y_A = -0.15
    tx = 0.1
    ty = 0.05
    gmsh.model.geo.addPoint(x_A, y_A, 0, hsize, 68)
    gmsh.model.geo.addPoint(x_A + tx, y_A - ty, 0, hsize, 69)
    gmsh.model.geo.addPoint(x_A + tx, y_A, 0, hsize, 70)
    gmsh.model.geo.addPoint(x_A, y_A + ty, 0, hsize, 71)
    gmsh.model.geo.addLine(68, 69, 55)
    gmsh.model.geo.addLine(69, 70, 56)
    gmsh.model.geo.addLine(70, 71, 57)
    gmsh.model.geo.addLine(71, 68, 58)

    gmsh.model.geo.addCurveLoop([55, 56, 57, 58], 4)

    ## TOP STRIE 1
    x_A = -0.95
    y_A = 0.15
    tx = 0.1
    ty = 0.05
    gmsh.model.geo.addPoint(x_A, y_A, 0, hsize, 72)
    gmsh.model.geo.addPoint(x_A + tx, y_A + ty, 0, hsize, 73)
    gmsh.model.geo.addPoint(x_A + tx, y_A, 0, hsize, 74)
    gmsh.model.geo.addPoint(x_A, y_A - ty, 0, hsize, 75)
    gmsh.model.geo.addLine(72, 73, 59)
    gmsh.model.geo.addLine(73, 74, 60)
    gmsh.model.geo.addLine(74, 75, 61)
    gmsh.model.geo.addLine(75, 72, 62)

    gmsh.model.geo.addCurveLoop([59, 60, 61, 62], 5)

    ## TOP STRIE 2
    x_A = -0.95
    y_A = 0.35
    tx = 0.1
    ty = 0.05
    gmsh.model.geo.addPoint(x_A, y_A, 0, hsize, 76)
    gmsh.model.geo.addPoint(x_A + tx, y_A + ty, 0, hsize, 77)
    gmsh.model.geo.addPoint(x_A + tx, y_A, 0, hsize, 78)
    gmsh.model.geo.addPoint(x_A, y_A - ty, 0, hsize, 79)
    gmsh.model.geo.addLine(76, 77, 63)
    gmsh.model.geo.addLine(77, 78, 64)
    gmsh.model.geo.addLine(78, 79, 65)
    gmsh.model.geo.addLine(79, 76, 66)

    gmsh.model.geo.addCurveLoop([63, 64, 65, 66], 6)

    ## BACK WING 
    gmsh.model.geo.addPoint(0.9, -0.1, 0, hsize, 80)
    gmsh.model.geo.addPoint(0.95, -0.1, 0, hsize, 81)
    gmsh.model.geo.addPoint(0.95, 0.1, 0, hsize, 82)
    gmsh.model.geo.addPoint(0.9, 0.1, 0, hsize, 83)
    gmsh.model.geo.addLine(80, 81, 67)
    gmsh.model.geo.addLine(81, 82, 68)
    gmsh.model.geo.addLine(82, 83, 69)
    gmsh.model.geo.addLine(83, 80, 70)

    gmsh.model.geo.addCurveLoop([67, 68, 69, 70], 7)

    # FULL DOMAIN
    gmsh.model.geo.addPlaneSurface([1, 2, 3, 4, 5, 6, 7], 1)

    gmsh.model.geo.synchronize()
    
    L2 = np.arange(49,71)

    # PHYSICAL GROUPS FOR BOUNDARY CONDITIONS
    gmsh.model.addPhysicalGroup(1, np.arange(1,71), tag_dirichlet)
    gmsh.model.setPhysicalName(1, tag_dirichlet, walls)

    gmsh.model.addPhysicalGroup(2, [1], 606)
    gmsh.model.setPhysicalName(2, 606, 'Surface')
    
    #########################################################################################################
    ############################### GENERATE THE MESH AND SAVING FILES ######################################
    #########################################################################################################

    # SI ON VEUT DES RECTANGLES PLUTOT
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