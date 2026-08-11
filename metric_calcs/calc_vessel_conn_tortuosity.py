# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:08:41 2026

@author: thomasnguyen
"""

import numpy as np
import pandas as pd


def calc_tortuosity(assignments,tree_1_vessels,tree_2_vessels):
    
    
    tortuosity_calcs = pd.DataFrame(index = range(len(tree_1_vessels)),
                                    columns = ['Tree 1 Vessel Number','Tree 2 Vessel Number',
                                               'Total Arclength','Total Curvature','Normalized Arclength',
                                               'Normalized Curvature','Geodesic Distance'])
    # Iterate through each vessel assignment pairing
    for vessel_count,vessel_assignment in enumerate(zip(*assignments)):
        total_arclength = 0
        total_curvature = 0
        geodesic_dist = np.linalg.norm(tree_2_vessels[vessel_count][0,0:3] - tree_1_vessels[vessel_count][0,0:3])
        tortuosity_calcs.loc[vessel_count,'Tree 1 Vessel Number'],tortuosity_calcs.loc[vessel_count,'Tree 2 Vessel Number'] = vessel_assignment
        tortuosity_calcs.loc[vessel_count,'Geodesic Distance'] = geodesic_dist
        # Iterate through each vessel that has been connected
            
            # T1_connection and #T2_connection are 9x7 arrays that contain info of the segment to make a connection
            # T1_connection[:,0:3] are the start points of a specific segment
            # T1_connection[:,3:6]are the end points of a specific segment
            # T1_connection[:,6] is the radius
        t1_connection = tree_1_vessels[vessel_count]
        t2_connection = tree_2_vessels[vessel_count]
        
        for vessel_half in [t1_connection,t2_connection]:
            
            p1 = vessel_half[:,0:3]
            p2 = vessel_half[:,3:6]
            
            vessel_coord = np.vstack([vessel_half[:,0:3],vessel_half[-1,3:6]])
            ds = np.linalg.norm(vessel_coord[1:] - vessel_coord[:-1],axis = 1)
            s = np.concatenate(([0], np.cumsum(ds)))
            dyds = np.gradient(vessel_coord,s,edge_order = 2,axis = 0)
            d2yds2 = np.gradient(dyds,s,edge_order = 2,axis = 0)
        
            vessel_curvature = np.trapezoid(np.linalg.norm(d2yds2,axis = 1),x = s)
            vessel_arclength = (np.linalg.norm(p2 - p1, axis=1)).sum()
            
            total_arclength += vessel_arclength
            total_curvature += vessel_curvature
        tortuosity_calcs.loc[vessel_count,'Total Arclength'] = total_arclength
        tortuosity_calcs.loc[vessel_count,'Total Curvature'] = total_curvature
        tortuosity_calcs.loc[vessel_count,'Normalized Arclength'] = total_arclength/geodesic_dist
        tortuosity_calcs.loc[vessel_count,'Normalized Curvature'] = total_curvature/geodesic_dist
    return tortuosity_calcs