# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 17:03:29 2026

@author: thomasnguyen
"""

import numpy as np
import pandas as pd
# def calc_vessel_angles(vessel_map,vessel_data):
#     """
#     This function calculates the dot product between a vessel and the tree root,
#     and the dot product between a vessel and the upstream vessel.
    
#     --- Inputs ---
        
#     vessel_map:             Vessel map that shows the upstream/downstream vessels
#                             for a given vessel. Obtained from tree.vessel_map
#     vessel_data:            NumPy array that contains all of the information
#                             for all vessels in a tree. Obtained from tree.data
                            
#     --- Returns ---
        
    
#     """
#     # Calculate tree root directional vector
#     dot_prod_parent = []
#     dot_prod_daughter = []
    
#     # Iterate through every vessel
#     for vessel_idx in vessel_map.keys():
        
#         # Check if there's downstream vessels
#         if len(vessel_map[vessel_idx]["downstream"]) > 0:
#             # Calculate upstream vessel directional vector
#             upstream_vessel_dir = vessel_data[vessel_idx,3:6] - vessel_data[vessel_idx,0:3]
#             upstream_vessel_dir /= np.linalg.norm(upstream_vessel_dir)
            
#             # Iterate through each downstream vessel
#             for ds_vessel_idx in vessel_map[vessel_idx]["downstream"]:
#                 # Calculate downstream vessel directional vector
#                 ds_vessel_dir = vessel_data[ds_vessel_idx,3:6] - vessel_data[ds_vessel_idx,0:3]
#                 ds_vessel_dir /= np.linalg.norm(ds_vessel_dir)
                
#                 dot_prod_1 = np.clip(np.dot(ds_vessel_dir,root_dir),-1,1)
#                 dot_prod_2 = np.clip(np.dot(ds_vessel_dir,upstream_vessel_dir),-1,1)
#                 dot_prod_root.append(dot_prod_1)
#                 dot_prod_adj.append(dot_prod_2)
#     return np.array(dot_prod_root),np.array(dot_prod_adj)


def calc_vessel_angles(vessel_map, vessel_data):
    # parent_to_continuation_angles = []
    # daughter_opening_angles = []
    # parent_to_terminal_angles = []
    
    
    angle_vessel_data = []
    for vessel_idx, info in vessel_map.items():
        # Identify direct children (vessels whose immediate parent is vessel_idx)
        direct_children = [
            ds_idx for ds_idx in info["downstream"]
            if len(vessel_map[ds_idx]["upstream"]) > 0 and vessel_map[ds_idx]["upstream"][-1] == vessel_idx
        ]
        
        # Evaluate bifurcations with at least 2 direct child vessels
        if len(direct_children) >= 2:
            # 1. Calculate normalized parent direction
            v_parent = vessel_data[vessel_idx, 3:6] - vessel_data[vessel_idx, 0:3]
            u_parent = v_parent / np.linalg.norm(v_parent)
            
            # 2. Calculate normalized child directions (taking the bifurcation pair)
            child_dirs = []
            for c_idx in direct_children[:2]:
                v_c = vessel_data[c_idx, 3:6] - vessel_data[c_idx, 0:3]
                child_dirs.append(v_c / np.linalg.norm(v_c))
            
            # 3. Identify continuation daughter vs terminal branch via dot product
            dot_0 = np.clip(np.dot(u_parent, child_dirs[0]), -1.0, 1.0)
            dot_1 = np.clip(np.dot(u_parent, child_dirs[1]), -1.0, 1.0)
            
            if dot_0 >= dot_1:
                u_continuation, u_terminal = child_dirs[0], child_dirs[1]
                cos_parent_cont = dot_0
                cont_vessel_idx = direct_children[0]
                term_vessel_idx = direct_children[1]
            else:
                u_continuation, u_terminal = child_dirs[1], child_dirs[0]
                cos_parent_cont = dot_1
                cont_vessel_idx = direct_children[1]
                term_vessel_idx = direct_children[0]
                
            # 4. Calculate daughter-to-daughter opening dot product
            cos_daughters = np.clip(np.dot(u_continuation, u_terminal), -1.0, 1.0)
            
            # 5. Caclculate parent to terminal dot product
            cos_parent_terminal = np.clip(np.dot(u_parent, u_terminal), -1.0, 1.0)
            
            # Store angles in degrees (bounded strictly between 0 and 180)
            # parent_to_continuation_angles.append(np.degrees(np.arccos(cos_parent_cont)))
            # daughter_opening_angles.append(np.degrees(np.arccos(cos_daughters)))
            # parent_to_terminal_angles.append(np.degrees(np.arccos(cos_parent_terminal)))
    # return np.array(parent_to_continuation_angles), np.array(parent_to_terminal_angles),np.array(daughter_opening_angles)
            
            angle_data = {"Parent Vessel Index": [vessel_idx],
                          "Continuation Daughter Vessel Index": cont_vessel_idx,
                          "Terminal Daughter Vessel Index": term_vessel_idx,
                          "Angle between Parent and Continuation Daughter": np.degrees(np.arccos(cos_parent_cont)),
                          "Angle between Parent and Terminal Daughter": np.degrees(np.arccos(cos_parent_terminal)),
                          "Angle between Daughers": np.degrees(np.arccos(cos_daughters))
                          }
            angle_vessel_data.append(angle_data)
            
    all_angle_data = pd.concat(
            [pd.DataFrame.from_dict(i) for i in angle_vessel_data],ignore_index = True)
    return all_angle_data
            
            
    
