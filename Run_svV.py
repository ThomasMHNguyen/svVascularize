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
    
# os.environ["OMP_NUM_THREADS"] = "1" 
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
#%%
import numpy as np
import random
import pyvista as pv
from svv.domain.domain import Domain
from svv.tree.tree import Tree
from svv.forest.forest import Forest
from svv.simulation.simulation import Simulation



#%% Test RNG for single tree

rand_seed = 800
# rng = np.random.default_rng(rand_seed)
# 1. Lock the global Python random module & NumPy's Global state
random.seed(rand_seed)       # Python's built-in random
np.random.seed(rand_seed)    # NumPy's global state


# Generate cube geometry
# domain = Domain(pv.read('biventricle_tissue.stl'))
domain = Domain(pv.Cube())

# Enforce same rng seed on domain
domain.set_random_seed(rand_seed)
domain.set_random_generator()

domain.create()
domain.solve()
domain.build()
tree = Tree()
tree.set_domain(domain)
tree.set_root()
tree.n_add(6)

# Visualizing the Tree and 

#%%
all_vessel_location_start = tree.data[:,0:3]
all_vessel_location_start = np.array([tree.data[0,0:3],
                                      tree.data[1,0:3],
                                      tree.data[1,3:6],
                                      tree.data[2,3:6],
                                      tree.data[3,3:6],
                                      tree.data[4,3:6],
                                      tree.data[5,0:3],
                                      tree.data[5,3:6],
                                      ])

vessel_label = ["Pt {}".format(i) for i in range(all_vessel_location_start.shape[0])]
test_poly = pv.PolyData(all_vessel_location_start)
test_poly['Vessel Labels'] = vessel_label

#%%
import pyvista as pv

plotter  = pv.Plotter()
plotter.add_mesh(tree.domain.boundary, color='grey', opacity=0.25)

# Plot each point and label
for i in range(tree.data.shape[0]):
    # for i in trange(tree.data.shape[0], desc='Building plot', unit='vessel', leave=False):
    center = (tree.data[i, 0:3] + tree.data[i, 3:6]) / 2
    direction = tree.data.get('w_basis', i)
    radius = tree.data.get('radius', i)
    length = tree.data.get('length', i)
    vessel = pv.Cylinder(center=center, direction=direction, radius=radius, height=length)
    plotter.add_mesh(vessel, color='red')
plotter.add_point_labels(test_poly, 'Vessel Labels', 
                         font_size=15,
                         point_size = 15,
                         always_visible = True,
                         shadow = False,
                         shape_opacity=0.5,
                         show_points = False)
plotter.show_axes()
plotter.show()
#%% Build 3 vessels

rand_seed = 800
# rng = np.random.default_rng(rand_seed)
# 1. Lock the global Python random module & NumPy's Global state
random.seed(rand_seed)       # Python's built-in random
np.random.seed(rand_seed)    # NumPy's global state


# Generate cube geometry
domain = Domain(pv.Cube())

# Enforce same rng seed on domain
domain.set_random_seed(rand_seed)
domain.set_random_generator()

domain.create()
domain.solve()
domain.build()



# 1. Define the 3 starting locations (Inlet 1, Inlet 2, Outlet 1)
# These should be points on the surface or just inside your 'cube' domain
pt_artery = np.array([0.3, 0.1, 0.5]) 
pt_portal = np.array([0.0, -0.1, 0.5]) 
pt_venous = np.array([-0.3, -0.2, -0.5])


# 2. Specify direction of growth of arteries/veins
dir_artery = np.array([0,0,-1])
dir_portal = np.array([0,0,-1])
dir_venous = np.array([0,0,1])

# 3. Feed in initial parameters to generate forest
n_networks = 1
n_trees_per_network = [3]
forest = Forest(
    domain =  domain,
    n_networks= n_networks, 
    n_trees_per_network= n_trees_per_network,
    physical_clearance = 0.05)

# 4. Append all starting points & directions into 1x3 array
all_start_points = [[pt_artery,pt_portal,pt_venous]]
all_directions = [[dir_artery,dir_portal,dir_venous]]

# 5. Set Domain and roots
forest.set_domain(domain)
forest.set_roots(
    start_points=all_start_points,
    directions= all_directions,
    )

# Enforce same rng seed on domain
for i in range(n_trees_per_network[0]):
    forest.networks[0][i].random_seed = rand_seed

# Add more vessesls and run connection routine              
forest.add(50)               
forest.connect()
forest.show(plot_domain = True)

# Save the complete forest (including connections)
# forest.save(os.path.join(script_dir,"vascular_network.forest"))

#%%




