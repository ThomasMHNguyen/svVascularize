import pyvista
from tqdm import trange

def show(tree, color='red', plot_domain=False, return_plotter=False, **kwargs):
    """
    Visualize a synthetic vascular tree using PyVista.

    This function creates a 3D visualization of a vascular tree, where each vessel is represented
    as a cylinder. The tree's data is iterated over to generate the cylinders based on their
    geometrical properties such as center, direction, radius, and length.

    Parameters
    ----------
    tree : object
        The synthetic vascular tree data structure. This object should have a `data` attribute,
        which is an array where each row represents a vessel. The `data` array is expected to have
        columns corresponding to the start and end points of the vessel (first three and next three
        columns, respectively). Additionally, the tree should provide access to methods like
        `get('w_basis', i)`, `get('radius', i)`, and `get('length', i)` that return the direction
        vector, radius, and length of each vessel, respectively.

    color : str, optional
        The color of the vessels in the visualization. Default is 'red'.

    plot_domain : bool, optional
        If True, the boundary of the domain in which the tree exists is also plotted with a
        semi-transparent grey mesh. Default is False.

    return_plotter : bool, optional
        If True, the PyVista `Plotter` object is returned instead of displaying the plot immediately.
        This can be useful for further customization of the plot outside of this function. Default
        is False.

    **kwargs : dict, optional
        Additional keyword arguments passed to the PyVista `Plotter` initializer. These can be used
        to control the appearance of the plot, such as the background color or window size.

    Returns
    -------
    plotter : pyvista.Plotter, optional
        If `return_plotter` is True, the PyVista `Plotter` object is returned. This allows further
        modifications to the plot before it is displayed.

    Notes
    -----
    The function uses a progress bar from `tqdm` to provide feedback during the plotting process.
    This can be useful when dealing with large trees, as it shows the progress of building the plot
    in real time.

    Examples
    --------

    .. code-block:: python

        >>> from svtoolkit import Tree, Domain
        >>> from svtoolkit.visualize.tree import show
        >>> import pyvista as pv
        >>> cube = Domain(pv.Cube()) # Create a cube domain from a PyVista mesh
        >>> cube.create() # Create the domain
        >>> cube.solve() # Solve the implicit functions of the domain
        >>> cube.build() # Build the domain
        >>> t = Tree() # Create a tree object
        >>> t.set_domain(cube) # Set the domain for the tree
        >>> t.set_root() # Set the root vessel
        >>> t.n_add(10) # Add 10 vessels to the tree
        >>> show(t, plot_domain=True) # Visualize the tree with the domain boundary

    """
    plotter = pyvista.Plotter(**kwargs)
    for i in trange(tree.data.shape[0], desc='Building plot', unit='vessel', leave=False):
        center = (tree.data[i, 0:3] + tree.data[i, 3:6]) / 2
        direction = tree.data.get('w_basis', i)
        radius = tree.data.get('radius', i)
        length = tree.data.get('length', i)
        vessel = pyvista.Cylinder(center=center, direction=direction, radius=radius, height=length)
        plotter.add_mesh(vessel, color=color)
    #Show Axes in 3D plot
    plotter.show_axes()
    plotter.show_grid()
    if plot_domain:
        plotter.add_mesh(tree.domain.boundary, color='grey', opacity=0.25)
    if return_plotter:
        return plotter
    else:
        plotter.show()

