# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:05:29 2026

@author: thomasnguyen
"""

import os, sys, time

### Point script to correct directory to ensure dependencies are correctly loaded ###
try:
    script_dir = os.path.abspath(os.path.join(
        os.path.dirname((os.path.abspath(__file__)))))

except NameError:
    # __file__ is not defined (e.g., in an interactive environment)
    # Use sys.argv[0] to get the script's name and then extract 1 directory above current one
    script_dir = os.path.abspath(os.path.join(
        os.path.abspath(sys.argv[0])))

# if script_dir not in sys.path:
#     sys.path.append(script_dir)   
if os.getcwd() != script_dir: 
    script_dir = os.path.join(script_dir)
    os.chdir(script_dir) 
    
import numpy as np
import random
import pyvista as pv
import pyvista
from svv.domain.domain import Domain
from svv.forest.forest import Forest

#%%

output_dir = 'C:/Users/thomasnguyen/OneDrive - University of California, San Diego Health/00_Projects/01a_SVVascularize/svVascularize/'
full_output_dir = os.path.join(output_dir,'Simulation_Results/Connection_Assign_Test/T_1_Anoosh/')
if not os.path.exists(full_output_dir):
    while True:
        try:
            os.makedirs(full_output_dir)
            break
        except OSError:
            continue

#%%

def plot_forest_annotate(forest,export_type):
    if export_type != 'save':
        plotter = pyvista.Plotter(notebook = False)
    elif export_type == 'save':
        plotter = pyvista.Plotter(notebook = False,off_screen=True)
    colors = ['red','blue']
    count = 0
    def _add_cylinder(p0, p1, radius, color, opacity=1.0):
        vec = p1 - p0
        length = np.linalg.norm(vec)
        if length <= 0:
            return
        direction = vec / length
        center = (p0 + p1) / 2
        cyl = pyvista.Cylinder(center=center, direction=direction, radius=radius, height=length)
        plotter.add_mesh(cyl, color=color, opacity=opacity)

    has_connections = getattr(forest, "connections", None) is not None and \
        getattr(forest.connections, "tree_connections", None)
    
    # for j in range(0,len(forest.networks[0])):
    #     for k in range(0,forest.networks[0][j].data.shape[0]):
    #         plotter.add_points(forest.networks[0][j].data[k,0:3], render_points_as_spheres=True, point_size=15,color = colors[j])
    #         plotter.add_points(forest.networks[0][j].data[k,3:6], render_points_as_spheres=True, point_size=15,color = colors[j])


    if has_connections:
        # Draw connected trees and connection vessels
        for net_idx, tree_conn in enumerate(forest.connections.tree_connections):
            for tree in tree_conn.connected_network:
                color = colors[count % len(colors)]
                for i in range(tree.data.shape[0]):
                    p0 = tree.data[i, 0:3]
                    p1 = tree.data[i, 3:6]
                    radius = tree.data.get('radius', i)
                    _add_cylinder(p0, p1, radius, color)
                count += 1

            # Connection vessels (between trees in this network)
            for tree_idx, vessel_list in enumerate(tree_conn.vessels):
                color = colors[tree_idx % len(colors)]
                for vessel in vessel_list:
                    for seg in vessel:
                        p0 = seg[0:3]
                        p1 = seg[3:6]
                        radius = seg[6]
                        _add_cylinder(p0, p1, radius, color)
    else:
        # Fall back to original visualization without connections
        for network in forest.networks:
            for tree in network:
                for i in range(tree.data.shape[0]):
                    center = (tree.data[i, 0:3] + tree.data[i, 3:6]) / 2
                    direction = tree.data.get('w_basis', i)
                    radius = tree.data.get('radius', i)
                    length = tree.data.get('length', i)
                    vessel = pyvista.Cylinder(center=center, direction=direction, radius=radius, height=length)
                    plotter.add_mesh(vessel, color=colors[count % len(colors)])
                count += 1
    # count = 0
    # # Fall back to original visualization without connections
    # for network in forest.networks:
    #     for tree in network:
    #         for i in range(tree.data.shape[0]):
    #             center = (tree.data[i, 0:3] + tree.data[i, 3:6]) / 2
    #             direction = tree.data.get('w_basis', i)
    #             radius = tree.data.get('radius', i)
    #             length = tree.data.get('length', i)
    #             vessel = pyvista.Cylinder(center=center, direction=direction, radius=radius, height=length)
    #             plotter.add_mesh(vessel, color=colors[count % len(colors)])
    #         count += 1
    #Show Axes in 3D plot
    plotter.show_axes()
    plotter.show_grid()
    plotter.add_mesh(forest.domain.boundary, color='grey', opacity=0.25)
    plotter.view_isometric()
    if export_type == 'display':
        plotter.show()
    elif export_type == 'save':
        plotter.screenshot(os.path.join(full_output_dir,'vessel_snapshot.jpg'))

#%% Generate Forest

forest_start_time = time.perf_counter()
rand_seed = 1000
rng = np.random.default_rng(rand_seed)
# 1. Lock the global Python random module & NumPy's Global state
random.seed(rand_seed)       # Python's built-in random
np.random.seed(rand_seed)    # NumPy's global state



# Creating the Tissue Domain
cube_mesh = pv.Cube(x_length = 1,y_length = 1,z_length = 1)
cube = Domain(cube_mesh)

# Enforce same rng seed on domain
cube.set_random_seed(rand_seed)
cube.set_random_generator()


cube.create()
cube.solve()
cube.build()

# Creating the Vascular Forest Object

# 1. Define the 3 starting locations (Inlet 1, Inlet 2, Outlet 1)
# These should be points on the surface or just inside your 'cube' domain
pt_artery = np.array([0, 0.50, 0]) 
pt_vein = np.array([0, -0.50, 0]) 



# 2. Specify direction of growth of arteries/veins
dir_artery = np.array([0,-1,0])
dir_vein = np.array([0,1,0])

# 3. Feed in initial parameters to generate forest
n_networks = 1
n_trees_per_network = [2]
forest = Forest(
    domain =  cube,
    n_networks= n_networks, 
    n_trees_per_network= n_trees_per_network,
    physical_clearance = 0.05,
        )
forest.set_domain(cube)

# 4. Append all starting points & directions into 1x3 array
all_start_points = [[pt_artery,pt_vein]]
all_directions = [[dir_artery,dir_vein]]
forest.set_roots(
    start_points=all_start_points,
    directions= all_directions,
    volume_fraction = 0.10,
    )
# forest.add(3)
forest.add(10,
           threshold = 0.25,
           volume_threshold = 0.3,
           n_points = 1000,
           n_closest_vessels = 100,
           initial_max_length = 0.5,
           vessels_max_angle = np.cos(np.deg2rad(75)))
forest.connect()
forest_end_time = time.perf_counter()
print("Time to generate and connect forests is: {:.3f} seconds".format(forest_end_time - forest_start_time))

#%%
plot_forest_annotate(forest,export_type = 'save')
# forest.show(plot_domain = True,notebook = False)


#%% Calculate vessel curvature
import pandas as pd


def calculate_arclength(forest):
    tree_vessel_information = []
    for net_idx, tree_conn in enumerate(forest.connections.tree_connections):
        # Connection vessels (between trees in this network)
        for tree_idx, vessel_list in enumerate(tree_conn.vessels):
            for vessel_idx,vessel in enumerate(vessel_list):
                p1 = vessel[:,0:3]
                p2 = vessel[:,3:6]
                
                vessel_coord = np.vstack([vessel[:,0:3],vessel[-1,3:6]])
                ds = np.linalg.norm(vessel_coord[1:] - vessel_coord[:-1],axis = 1)
                s = np.concatenate(([0], np.cumsum(ds)))
                dyds = np.gradient(vessel_coord,s,edge_order = 2,axis = 0)
                d2yds2 = np.gradient(dyds,s,edge_order = 2,axis = 0)
                
                total_curvature = np.trapezoid(np.linalg.norm(d2yds2,axis = 1),x = s)
                vessel_geo_length = np.linalg.norm(vessel[-1,3:6] - vessel[0,0:3])
                vessel_arclength = (np.linalg.norm(p2 - p1, axis=1)).sum()
                

                vessel_info = {"Random Seed": rand_seed,
                                "Tree Index": tree_idx,
                               "Vessel Index": vessel_idx,
                               "Vessel Arc Length": vessel_arclength,
                               "Vessel Geodesic Length": vessel_geo_length,
                               "Vessel Tortuousity Index": float(vessel_arclength/vessel_geo_length),
                               "Vessel Curvature": [total_curvature],
                               "Vessel Radius": vessel[0,6],
                               "Method": "Modified"}
                tree_vessel_information.append(vessel_info)
    tree_vessel_info = pd.concat(
    [pd.DataFrame.from_dict(i) for i in tree_vessel_information],ignore_index = True)    
    return tree_vessel_info

forest_vessel_info = calculate_arclength(forest)


#%% Test changes for new Pyvista version - build meshes
# from svv.simulation.simulation import Simulation
# import pandas as pd


# sim_start_time = time.perf_counter()

# sim = Simulation(forest,name = 'Simulation_Results/Chloe_Structures/T_2/',directory = output_dir)

# cap_resolution = 10
# # Build All Surface/Volume Meshes for 3D CFD Simulation - Return network mesh
# # sim.build_meshes(boundary_layer = False,cap_resolution = cap_resolution)
# sim_end_time = time.perf_counter()
# print("Time to generate meshes is: {0:.3f} seconds for a cap resolution of {1:2d}.".format((sim_end_time - sim_start_time),cap_resolution))
#%%

import pandas as pd
# if hasattr(sim, 'fluid_domain_surface_meshes') and sim.fluid_domain_surface_meshes:
#     for i, mesh in enumerate(sim.fluid_domain_surface_meshes):
#         filename = os.path.join(full_output_dir, f"surface_mesh_{i}.stl")
#         mesh.save(filename)
#         print(f"Saved: {filename}")

# # Save volume meshes (.vtk)
# if hasattr(sim, 'fluid_domain_volume_meshes') and sim.fluid_domain_volume_meshes:
#     for i, mesh in enumerate(sim.fluid_domain_volume_meshes):
#         filename = os.path.join(full_output_dir, f"volume_mesh_{i}.vtk")  # or .vtu
#         mesh.save(filename)
#         print(f"Saved: {filename}")

# params_df = pd.DataFrame(index = str(range(0,1)),columns = ['RNG Number'])
# params_df.loc['0','RNG Number'] = rand_seed
# params_df.to_csv(os.path.join(output_dir,'rng_params.csv'))

forest_vessel_info.to_csv(os.path.join(full_output_dir,'conn_assign_results_original.csv'))