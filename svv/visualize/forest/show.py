import numpy as np
import pyvista


def show(forest, plot_domain=False, return_plotter=False, **kwargs):
    """
    Visualize a forest, optionally including forest connection networks when present.
    """
    colors = kwargs.get('colors', ['red', 'blue', 'green', 'yellow', 'purple',
                                   'orange', 'cyan', 'magenta', 'white', 'black'])
    plotter = pyvista.Plotter(**kwargs)
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

    #Show Axes in 3D plot
    plotter.show_axes()
    plotter.show_grid()
    if plot_domain:
        plotter.add_mesh(forest.domain.boundary, color='grey', opacity=0.25)
    if return_plotter:
        return plotter
    else:
        plotter.show()
