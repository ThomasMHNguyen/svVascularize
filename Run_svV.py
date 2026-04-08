# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 09:49:07 2026

@author: thomasnguyen
"""

import os, sys

### Point script to correct directory to ensure dependencies are correctly loaded ###
try:
    script_dir = os.path.abspath(os.path.join(
        os.path.dirname((os.path.abspath(__file__))),'..'))

except NameError:
    # __file__ is not defined (e.g., in an interactive environment)
    # Use sys.argv[0] to get the script's name and then extract 1 directory above current one
    script_dir = os.path.abspath(os.path.join(
        os.path.abspath(sys.argv[0]),'..'))

# if script_dir not in sys.path:
#     sys.path.append(script_dir)   
if os.getcwd() != script_dir: 
    script_dir = os.path.join(script_dir,'svVascularize')
    os.chdir(script_dir) 
    
#%%
import pyvista as pv
from svv.domain.domain import Domain
from svv.tree.tree import Tree
from svv.forest.forest import Forest
from svv.simulation.simulation import Simulation
import numpy as np

#%% Build 3 vessels

# Generate cube geometry
domain = Domain(pv.Cube())
domain.create()
domain.solve()
domain.build()



# 1. Define the 3 starting locations (Inlet 1, Inlet 2, Outlet 1)
# These should be points on the surface or just inside your 'cube' domain
pt_artery = np.array([0.3, 0.1, 0.5]) 
pt_portal = np.array([0.0, -0.1, 0.5]) 
pt_venous = np.array([-0.3, -0.2, -0.5])

dir_artery = np.array([0,0,-1])
dir_portal = np.array([0,0,-1])
dir_venous = np.array([0,0,1])

# init_config = {
#     "domain": cube,
#     "n_networks": 1, 
#     "n_trees_per_network": [3], # One tree for each network
#     # "domain_clearance": 0.05,
#     # "start_points": [[pt_artery]],
#     # "start_points": [[pt_artery], [pt_portal], [pt_venous]],
#     # "directions": [[dir_in], [dir_in], [dir_out]],
#     "physical_clearance": 0.05, # Distance maintained between different networks
#     # "compete": True             # Crucial for interlacing the networks
# }

n_networks = 1
n_trees_per_network = [3]
forest = Forest(
    domain =  domain,
    n_networks= n_networks, 
    n_trees_per_network= n_trees_per_network,
    physical_clearance = 0.05)


all_start_points = [[pt_artery,pt_portal,pt_venous]]
all_directions = [[dir_artery,dir_portal,dir_venous]]

for i in range(n_networks):
    for j in range(n_trees_per_network[i]):
        try:
            print(all_start_points[i][j])
        except IndexError:
            print("Error with i = {}, j = {}".format(i,j))

forest.set_domain(domain)
forest.set_roots(
    start_points=all_start_points,
    directions= [[dir_artery,dir_portal,dir_venous]]
    )
                
                
forest.add(2) #number of vessels per tree
# forest.connect()
forest.show(plot_domain = True)


