# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:40:35 2026

@author: thomasnguyen
"""

import pyvista as pv
from svv.domain.domain import Domain
from svv.tree.tree import Tree
from svv.forest.forest import Forest
from svv.simulation.simulation import Simulation
import numpy as np

#%%

# Prepare a domain and build its implicit representation
domain = Domain(pv.Cube())
domain.create()
domain.solve()
domain.build()

# Initialize a forest with two trees in one network
trees_per_network = [4]
forest = Forest(domain=domain, n_networks=1, n_trees_per_network=trees_per_network)
forest.set_domain(domain)
forest.set_roots()  # Randomize root placement
forest.add(3)      # Grow 50 vessels per tree

# Connect the trees (starting with cubic curves)
forest.connect(4)

forest.show(plot_domain=True)
#%%
# Prepare a domain and build its implicit representation
domain = Domain(pv.Cube())
domain.create()
domain.solve()
domain.build()

# Initialize a forest with two trees in one network
trees_per_network = [3,2]
forest = Forest(domain=domain, n_networks=2, n_trees_per_network=trees_per_network)
forest.set_domain(domain)
forest.set_roots()  # Randomize root placement
forest.add(3)      # Grow 50 vessels per tree

# Connect the trees (starting with cubic curves)
forest.connect(4)

forest.show(plot_domain=True)