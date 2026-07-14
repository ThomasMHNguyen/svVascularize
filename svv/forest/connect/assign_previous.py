import numpy

from tqdm import trange, tqdm
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import splprep, splev, interp1d
from copy import deepcopy
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

def assign_network(forest, *args, **kwargs):
    """
    Assign the terminal connections among tree objects within a
    forest network. The assignment is based on the minimum distance
    between terminal points of the tree.

    Parameters
    ----------
    forest : svtoolkit.forest.Forest
        A forest object that contains a collection of trees.
    args : int (optional)
        The index of the network to be assigned.

    Returns
    -------
    network_assignments : list of list of int
        A list of terminal indices for each tree in the network.
    network_connections : list of list of functions
        A list of functions that define the connection between
        terminal points of the trees in the network. By default,
        the connection among n interpenetrating trees is defined
        by the midpoint (t=0.5) of spline curve that assigns the first
        two trees in the network.
    kwargs : dict
        Additional keyword arguments.
        Keyword arguments include:
            t : float
                The parameter value for the connection point among
                interpenetrating trees. By default, this is defined
                as the midpoint (t=0.5).
    """
    network_connections = []
    network_assignments = []
    t = kwargs.get('t', 0.5)
    show = kwargs.get('show', False)
    if len(args) == 0:
        network_id = 0
    else:
        network_id = args[0]
    neighbors = kwargs.get('neighbors', int(t * numpy.sum(numpy.all(numpy.isnan(forest.networks[network_id][0].data[:, 15:17]), axis=1))))
    if forest.n_trees_per_network[network_id] >= 2:
        tree_0 = forest.networks[network_id][0].data
        tree_1 = forest.networks[network_id][1].data
        idx_0 = numpy.argwhere(numpy.all(numpy.isnan(tree_0[:, 15:17]), axis=1)).flatten()
        idx_1 = numpy.argwhere(numpy.all(numpy.isnan(tree_1[:, 15:17]), axis=1)).flatten()
        terminals_0_ind = idx_0
        terminals_0_pts = tree_0[idx_0, 3:6]
        terminals_0_tree = cKDTree(terminals_0_pts)
        terminals_1_ind = idx_1
        terminals_1_pts = tree_1[idx_1, 3:6]
        terminals_1_tree = cKDTree(terminals_1_pts)
        neighbors = min(neighbors, terminals_0_pts.shape[0], terminals_1_pts.shape[0])
        rows = numpy.repeat(numpy.arange(terminals_0_pts.shape[0]), neighbors)
        cols = numpy.repeat(numpy.arange(terminals_1_pts.shape[0]), neighbors)
        network_assignments.append(terminals_0_ind.tolist())
        #C = numpy.zeros((terminals_0_pts.shape[0], terminals_1_pts.shape[0]))
        C = numpy.full((terminals_0_pts.shape[0], terminals_1_pts.shape[0]), 1e8)
        # Sparse equivalent of C

        #M = [[[None]]*terminals_0_pts.shape[0]]*terminals_1_pts.shape[0]
        M_sparse = []
        if forest.convex:
            #C = cdist(terminals_0_pts, terminals_1_pts)
            dists_1, idxs_1 = terminals_1_tree.query(terminals_0_pts, k=neighbors)
            dists_0, idxs_0 = terminals_0_tree.query(terminals_1_pts, k=neighbors)
            C[rows, idxs_1.flatten()] = dists_1.flatten()
            C[idxs_0.flatten(), cols] = dists_0.flatten()
            all_rows = numpy.array(rows.tolist() + idxs_0.flatten().tolist())
            all_cols = numpy.array(idxs_1.flatten().tolist() + cols.tolist())
            all_data = numpy.array(dists_1.flatten().tolist() + dists_0.flatten().tolist())
            function_data = []
            for i, j in zip(all_rows, all_cols):
                path_pts = deepcopy(numpy.vstack((terminals_0_pts[i, :], terminals_1_pts[j, :])))
                k = 1
                tck = deepcopy(splprep(path_pts.T, s=0, k=k))
                def func(t_, tck=tck):
                    return numpy.array(splev(t_, tck[0])).T
                function_data.append(func)
        else:
            # This matches the terminals of the second tree to queried terminals of the first tree
            dists_1, idxs_1 = terminals_1_tree.query(terminals_0_pts, k=neighbors)
            # This matches the terminals of the first tree to queried terminals of the second tree
            dists_0, idxs_0 = terminals_0_tree.query(terminals_1_pts, k=neighbors)
            # Calculate the rectangular distance matrix between the terminal points
            # M = [[[None]] * terminals_0_pts.shape[0]] * terminals_1_pts.shape[0]
            all_rows = numpy.array(rows.tolist() + idxs_0.flatten().tolist())
            all_cols = numpy.array(idxs_1.flatten().tolist() + cols.tolist())
            all_data = []
            function_data = []
            # M_sparse
            for i, j in tqdm(zip(all_rows, all_cols), total=len(all_rows), desc='Calculating geodesics', leave=False):
                path_pts = numpy.vstack((terminals_0_pts[i, :], terminals_1_pts[j, :]))
                k = 1
                tck = splprep(path_pts.T, s=0, k=k)
                t = numpy.linspace(0, 1, path_pts.shape[0])
                xpts = interp1d(t, path_pts[:, 0])
                ypts = interp1d(t, path_pts[:, 1])
                zpts = interp1d(t, path_pts[:, 2])
                def func(t_, xpts=xpts, ypts=ypts, zpts=zpts):
                    return numpy.array([xpts(t_), ypts(t_), zpts(t_)]).T
                pts = func(numpy.linspace(0, 1))
                values = forest.domain(pts)
                dists = numpy.linalg.norm(numpy.diff(pts, axis=0), axis=1)
                if numpy.any(values > 0):
                    path, dists, _ = forest.geodesic(terminals_0_pts[i, :], terminals_1_pts[j, :])
                    path_pts = forest.domain.mesh.points[path, :]
                    k = 1
                    tck = splprep(path_pts.T, s=0, k=k)
                    t = numpy.linspace(0, 1, path_pts.shape[0])
                    xpts = interp1d(t, path_pts[:, 0])
                    ypts = interp1d(t, path_pts[:, 1])
                    zpts = interp1d(t, path_pts[:, 2])
                    def func(t_, xpts=xpts, ypts=ypts, zpts=zpts):
                        return numpy.array([xpts(t_), ypts(t_), zpts(t_)]).T
                #M[i][idxs_1[i, j]] = func
                #C[i, idxs_1[i, j]] = numpy.sum(dists)
                all_data.append(numpy.sum(dists))
                function_data.append(func)
            """
            for i in trange(idxs_0.shape[0], desc='Calculating geodesics II', leave=False):
                for j in range(idxs_0.shape[1]):
                    if not isinstance(M[idxs_0[i, j]][i], type(None)):
                        continue
                    path_pts = numpy.vstack((terminals_0_pts[idxs_0[i, j], :], terminals_1_pts[i, :]))
                    k = 1
                    tck = splprep(path_pts.T, s=0, k=k)
                    t = numpy.linspace(0, 1, path_pts.shape[0])
                    xpts = interp1d(t, path_pts[:, 0])
                    ypts = interp1d(t, path_pts[:, 1])
                    zpts = interp1d(t, path_pts[:, 2])
                    def func(t_, xpts=xpts, ypts=ypts, zpts=zpts):
                        return numpy.array([xpts(t_), ypts(t_), zpts(t_)]).T
                    pts = func(numpy.linspace(0, 1))
                    values = forest.domain(pts)
                    dists = numpy.linalg.norm(numpy.diff(pts, axis=0), axis=1)
                    if numpy.any(values > 0):
                        path, dists, _ = forest.geodesic(terminals_0_pts[idxs_0[i, j], :], terminals_1_pts[i, :])
                        path_pts = forest.domain.mesh.points[path, :]
                        k = 1
                        tck = splprep(path_pts.T, s=0, k=k)
                        t = numpy.linspace(0, 1, path_pts.shape[0])
                        xpts = interp1d(t, path_pts[:, 0])
                        ypts = interp1d(t, path_pts[:, 1])
                        zpts = interp1d(t, path_pts[:, 2])
                        def func(t_, xpts=xpts, ypts=ypts, zpts=zpts):
                            return numpy.array([xpts(t_), ypts(t_), zpts(t_)]).T
                    M[idxs_0[i, j]][i] = func
                    C[idxs_0[i, j], i] = numpy.sum(dists)
            """
            """
            for i in trange(terminals_0_pts.shape[0], desc='Calculating geodesics', leave=False):
                tmp_M = []
                for j in range(terminals_1_pts.shape[0]):
                    path_pts = numpy.vstack((terminals_0_pts[i, :], terminals_1_pts[j, :]))
                    k = 1
                    tck = splprep(path_pts.T, s=0, k=k)
                    t = numpy.linspace(0, 1, path_pts.shape[0])
                    xpts = interp1d(t, path_pts[:, 0])
                    ypts = interp1d(t, path_pts[:, 1])
                    zpts = interp1d(t, path_pts[:, 2])
                    #func = lambda t_: numpy.array(splev(t_, tck[0])).T
                    #func = lambda t_: numpy.array([xpts(t_), ypts(t_), zpts(t_)]).T
                    def func(t_, xpts=xpts, ypts=ypts, zpts=zpts):
                        return numpy.array([xpts(t_), ypts(t_), zpts(t_)]).T
                    pts = func(numpy.linspace(0, 1))
                    values = forest.domain(pts)
                    dists = numpy.linalg.norm(numpy.diff(pts, axis=0), axis=1)
                    if numpy.any(values > 0):
                        path, dists, _ = forest.geodesic(terminals_0_pts[i, :], terminals_1_pts[j, :])
                        path_pts = forest.domain.mesh.points[path, :]
                        #geodesic_generator = lambda data: geodesic(data, start=terminals_0_pts[i, :],
                        #                                           end=terminals_1_pts[j, :])
                        #cost = lambda data: geodesic_cost(data, curve_generator=geodesic_generator,
                        #                                  boundary_func=forest.domain.evaluate)
                        #res = minimize(cost, path_pts, method="L-BFGS-B")
                        #path_pts = res.x.reshape(-1, 3)
                        #dists = numpy.linalg.norm(numpy.diff(path_pts, axis=0), axis=1)
                        #C[i, j] = numpy.sum(dists)
                        # TODO: add check that the terminal points are not a node point
                        path_pts = numpy.vstack((terminals_0_pts[i, :], path_pts, terminals_1_pts[j, :]))
                        #if path_pts.shape[0] > 3:
                        #    k = 3
                        #elif path_pts.shape[0] > 2:
                        #    k = 2
                        #else:
                        #    k = 1
                        #tck = splprep(path_pts.T, s=0, k=k)
                        t = numpy.linspace(0, 1, path_pts.shape[0])
                        xpts = interp1d(t, path_pts[:, 0])
                        ypts = interp1d(t, path_pts[:, 1])
                        zpts = interp1d(t, path_pts[:, 2])
                        def func(t_, xpts=xpts, ypts=ypts, zpts=zpts):
                            return numpy.array([xpts(t_), ypts(t_), zpts(t_)]).T
                        #func = lambda t_: numpy.array([xpts(t_), ypts(t_), zpts(t_)]).T
                        #func = lambda t: numpy.array(splev(t, tck[0])).T
                    tmp_M.append(func)
                    C[i, j] = numpy.sum(dists)
                    #M[i][j] = func
                M.append(tmp_M)
            """
        C_sparse = coo_matrix((all_data, (all_rows, all_cols)),
                              shape=(terminals_0_pts.shape[0], terminals_1_pts.shape[0]))
        #M_dense = numpy.full((terminals_0_pts.shape[0], terminals_1_pts.shape[0]), None)
        M_sparse = {}
        for i, j, func in zip(all_rows, all_cols, function_data):
            M_sparse[str(i)+','+str(j)] = func
        function_data = numpy.array(function_data)
        print("Function data shape: ", len(M_sparse))
        #M_dense[all_rows, all_cols] = function_data
        #M_sparse = coo_matrix((function_data, (all_rows, all_cols)),
        #                      shape=(terminals_0_pts.shape[0], terminals_1_pts.shape[0]))
        print("Calculating optimal assignment...")
        #_, assignment = linear_sum_assignment(C)
        try:
            row_ind, col_ind = min_weight_full_bipartite_matching(C_sparse)
        except:
            print("ERROR: Could not find optimal assignment. Try increasing the number of neighbors allowed in search.")
            return None, None
        print("Finished.")
        midpoints = []
        for i, j in zip(row_ind, col_ind):
            m_val = M_sparse[str(i)+','+str(j)]
            if isinstance(m_val, type(None)):
                print("ERROR: SHOULD NOT BE NONE")
            midpoints.append(m_val)
        network_assignments.append(terminals_1_ind[col_ind].tolist())
        network_connections.append(midpoints)
        # [TODO] Remove this block of code since the geodesics or linear connections will have to be re-calculated
        """
        if forest.n_trees_per_network[network_id] > 2:
            mid = numpy.array([midpoints[i](t) for i in range(len(midpoints))])
            for N in range(2, forest.n_trees_per_network[network]):
                tree_n = forest.networks[network][N].data
                idx_n = numpy.argwhere(numpy.all(numpy.isnan(tree_n[:, 15:17]), axis=1)).flatten()
                terminals_n_ind = idx_n
                terminals_n_pts = tree_n[idx_n, 3:6]
                C = numpy.zeros((mid.shape[0], terminals_1_pts.shape[0]))
                MN = [[[None]] * mid.shape[0]] * terminals_n_pts.shape[0]
                if forest.convex:
                    C = cdist(mid,terminals_n_pts)
                    for i in range(mid.shape[0]):
                        for j in range(terminals_n_pts.shape[0]):
                            path_pts = numpy.vstack((mid[i, :], terminals_n_pts[j, :]))
                            k = 1
                            tck = splprep(path_pts.T, s=0, k=k)
                            MN[i][j] = lambda t_: numpy.array(splev(t_, tck[0])).T
                else:
                    for i in range(mid.shape[0]):
                        for j in range(terminals_n_pts.shape[0]):
                            path, dists, _ = forest.geodesic(mid[i, :], terminals_n_pts[j, :])
                            C[i, j] = numpy.sum(dists)
                            path_pts = forest.domain.mesh.points[path, :]
                            # TODO: add check that the terminal points are not a node point
                            path_pts = numpy.vstack((terminals_0_pts[i, :], path_pts, terminals_1_pts[j, :]))
                            if path_pts.shape[0] > 3:
                                k = 3
                            elif path_pts.shape[0] > 2:
                                k = 2
                            else:
                                k = 1
                            tck = splprep(path_pts.T, s=0, k=k)
                            MN[i][j] = lambda t: numpy.array(splev(t, tck[0])).T
                _, assignment = linear_sum_assignment(C)
                midpoints_n = [MN[i][j] for i, j in enumerate(assignment)]
                network_assignments.append(terminals_n_ind[assignment].tolist())
                network_connections.extend([midpoints_n])
        """
    return network_assignments, network_connections


def assign_network_vector(forest, network_id, midpoints, **kwargs):
    """
    Assign the terminal connections among tree objects within a
    forest network. The assignment is based on the minimum distance
    between terminal points of the tree.

    Parameters
    ----------
    forest : svtoolkit.forest.Forest
        A forest object that contains a collection of trees.
    args : int (optional)
        The index of the network to be assigned.

    Returns
    -------
    network_assignments : list of list of int
        A list of terminal indices for each tree in the network.
    network_connections : list of list of functions
        A list of functions that define the connection between
        terminal points of the trees in the network. By default,
        the connection among n interpenetrating trees is defined
        by the midpoint (t=0.5) of spline curve that assigns the first
        two trees in the network.
    kwargs : dict
        Additional keyword arguments.
        Keyword arguments include:
            t : float
                The parameter value for the connection point among
                interpenetrating trees. By default, this is defined
                as the midpoint (t=0.5).
    """
    network_connections = []
    network_assignments = []
    neighbors = kwargs.get('neighbors', 5)
    if forest.n_trees_per_network[network_id] > 2:
        for N in range(2, forest.n_trees_per_network[network_id]):
            tree_n = forest.networks[network_id][N].data
            idx_n = numpy.argwhere(numpy.all(numpy.isnan(tree_n[:, 15:17]), axis=1)).flatten()
            terminals_n_ind = idx_n
            terminals_n_pts = tree_n[idx_n, 3:6]
            neighbors = min(neighbors, midpoints.shape[0], terminals_n_pts.shape[0])
            terminals_n_tree = cKDTree(terminals_n_pts)
            midpoints_tree = cKDTree(midpoints)
            #C = numpy.zeros((midpoints.shape[0], terminals_n_pts.shape[0]))
            C = numpy.full((midpoints.shape[0], terminals_n_pts.shape[0]), 1e8)
            MN = [[[None]] * midpoints.shape[0]] * terminals_n_pts.shape[0]
            if forest.convex:
                #C = cdist(midpoints, terminals_n_pts)
                dists_1, idxs_1 = midpoints_tree.query(terminals_n_pts, k=neighbors)
                dists_0, idxs_0 = terminals_n_tree.query(midpoints, k=neighbors)
                rows = numpy.repeat(numpy.arange(terminals_n_pts.shape[0]), neighbors)
                cols = numpy.repeat(numpy.arange(midpoints.shape[0]), neighbors)
                C[cols, idxs_1.flatten()] = dists_1.flatten()
                C[idxs_0.flatten(), rows] = dists_0.flatten()
                for i in range(midpoints.shape[0]):
                    for j in range(terminals_n_pts.shape[0]):
                        path_pts = numpy.vstack((terminals_n_pts[j, :], midpoints[i, :]))
                        k = 1
                        tck = splprep(path_pts.T, s=0, k=k)
                        def func(t_, tck=tck):
                            return numpy.array(splev(t_, tck[0])).T
                        #MN[i][j] = lambda t_: numpy.array(splev(t_, tck[0])).T
                        MN[i][j] = deepcopy(func)
            else:
                dists_1, idxs_1 = midpoints_tree.query(terminals_n_pts, k=neighbors)
                dists_0, idxs_0 = terminals_n_tree.query(midpoints, k=neighbors)
                for i in trange(idxs_1.shape[0], desc='Calculating geodesics I', leave=False):
                    for j in range(idxs_1.shape[1]):
                        path_pts = numpy.vstack((terminals_n_pts[i, :], midpoints[idxs_1[i, j], :]))
                        k = 1
                        tck = splprep(path_pts.T, s=0, k=k)
                        t = numpy.linspace(0, 1, path_pts.shape[0])
                        xpts = interp1d(t, path_pts[:, 0])
                        ypts = interp1d(t, path_pts[:, 1])
                        zpts = interp1d(t, path_pts[:, 2])

                        def func(t_, xpts=xpts, ypts=ypts, zpts=zpts):
                            return numpy.array([xpts(t_), ypts(t_), zpts(t_)]).T

                        pts = func(numpy.linspace(0, 1))
                        values = forest.domain(pts)
                        dists = numpy.linalg.norm(numpy.diff(pts, axis=0), axis=1)
                        if numpy.any(values > 0):
                            path, dists, _ = forest.geodesic(terminals_n_pts[i, :], midpoints[idxs_1[i, j], :])
                            path_pts = forest.domain.mesh.points[path, :]
                            k = 1
                            tck = splprep(path_pts.T, s=0, k=k)
                            t = numpy.linspace(0, 1, path_pts.shape[0])
                            xpts = interp1d(t, path_pts[:, 0])
                            ypts = interp1d(t, path_pts[:, 1])
                            zpts = interp1d(t, path_pts[:, 2])

                            def func(t_, xpts=xpts, ypts=ypts, zpts=zpts):
                                return numpy.array([xpts(t_), ypts(t_), zpts(t_)]).T
                        MN[i][idxs_1[i, j]] = func
                        C[i, idxs_1[i, j]] = numpy.sum(dists)
                for i in trange(idxs_0.shape[0], desc='Calculating geodesics II', leave=False):
                    for j in range(idxs_0.shape[1]):
                        if not isinstance(MN[idxs_0[i, j]][i], type(None)):
                            continue
                        path_pts = numpy.vstack((terminals_n_pts[idxs_0[i, j], :], midpoints[i, :]))
                        k = 1
                        tck = splprep(path_pts.T, s=0, k=k)
                        t = numpy.linspace(0, 1, path_pts.shape[0])
                        xpts = interp1d(t, path_pts[:, 0])
                        ypts = interp1d(t, path_pts[:, 1])
                        zpts = interp1d(t, path_pts[:, 2])

                        def func(t_, xpts=xpts, ypts=ypts, zpts=zpts):
                            return numpy.array([xpts(t_), ypts(t_), zpts(t_)]).T

                        pts = func(numpy.linspace(0, 1))
                        values = forest.domain(pts)
                        dists = numpy.linalg.norm(numpy.diff(pts, axis=0), axis=1)
                        if numpy.any(values > 0):
                            path, dists, _ = forest.geodesic(terminals_n_pts[idxs_0[i, j], :], midpoints[i, :])
                            path_pts = forest.domain.mesh.points[path, :]
                            k = 1
                            tck = splprep(path_pts.T, s=0, k=k)
                            t = numpy.linspace(0, 1, path_pts.shape[0])
                            xpts = interp1d(t, path_pts[:, 0])
                            ypts = interp1d(t, path_pts[:, 1])
                            zpts = interp1d(t, path_pts[:, 2])

                            def func(t_, xpts=xpts, ypts=ypts, zpts=zpts):
                                return numpy.array([xpts(t_), ypts(t_), zpts(t_)]).T
                        MN[idxs_0[i, j]][i] = func
                        C[idxs_0[i, j], i] = numpy.sum(dists)
                """
                for i in trange(midpoints.shape[0], desc='Calculating geodesics for tree: {}'.format(N), leave=False):
                    for j in range(terminals_n_pts.shape[0]):
                        path_pts = numpy.vstack((terminals_n_pts[i, :], midpoints[j, :]))
                        k = 1
                        tck = splprep(path_pts.T, s=0, k=k)
                        func = lambda t_: numpy.array(splev(t_, tck[0])).T
                        pts = func(numpy.linspace(0, 1))
                        values = forest.domain(pts)
                        dists = numpy.linalg.norm(numpy.diff(pts, axis=0), axis=1)
                        if numpy.any(values > 0):
                            path, dists, _ = forest.geodesic(terminals_n_pts[i, :], midpoints[j, :])
                            path_pts = forest.domain.mesh.points[path, :]
                            # geodesic_generator = lambda data: geodesic(data, start=terminals_0_pts[i, :],
                            #                                           end=terminals_1_pts[j, :])
                            # cost = lambda data: geodesic_cost(data, curve_generator=geodesic_generator,
                            #                                  boundary_func=forest.domain.evaluate)
                            # res = minimize(cost, path_pts, method="L-BFGS-B")
                            # path_pts = res.x.reshape(-1, 3)
                            # dists = numpy.linalg.norm(numpy.diff(path_pts, axis=0), axis=1)
                            # C[i, j] = numpy.sum(dists)
                            # TODO: add check that the terminal points are not a node point
                            path_pts = numpy.vstack((terminals_n_pts[i, :], path_pts, midpoints[j, :]))
                            if path_pts.shape[0] > 3:
                                k = 3
                            elif path_pts.shape[0] > 2:
                                k = 2
                            else:
                                k = 1
                            tck = splprep(path_pts.T, s=0, k=k)
                            func = lambda t: numpy.array(splev(t, tck[0])).T
                        C[i, j] = numpy.sum(dists)
                        MN[i][j] = func
                        
                        path, dists, _ = forest.geodesic(terminals_n_pts[i, :], midpoints[j, :])
                        C[i, j] = numpy.sum(dists)
                        path_pts = forest.domain.mesh.points[path, :]
                        # TODO: add check that the terminal points are not a node point
                        path_pts = numpy.vstack((terminals_n_pts[i, :], path_pts, midpoints[j, :]))
                        if path_pts.shape[0] > 3:
                            k = 3
                        elif path_pts.shape[0] > 2:
                            k = 2
                        else:
                            k = 1
                        tck = splprep(path_pts.T, s=0, k=k)
                        MN[i][j] = lambda t: numpy.array(splev(t, tck[0])).T
                        
            """
            _, assignment = linear_sum_assignment(C)
            #midpoints_n = [MN[i][j] for i, j in enumerate(assignment)]
            midpoints_n = []
            for i, j in enumerate(assignment):
                midpoints_n.append(MN[i][j])
            network_assignments.append(terminals_n_ind[assignment].tolist())
            network_connections.extend([midpoints_n])
    return network_assignments, network_connections


def assign(forest, **kwargs):
    """
    Assign the terminal connections among tree objects within all
    forest networks. The assignment is based on the minimum distance
    between terminal points of the tree.

    Parameters
    ----------
    forest : svtoolkit.forest.Forest
        A forest object that contains a collection of trees.
    kwargs : dict
        Additional keyword arguments.
        Keyword arguments include:
            t : float
                The parameter value for the connection point among
                interpenetrating trees. By default, this is defined
                as the midpoint (t=0.5).

    Returns
    -------
    assignments : list of list of list of int
        A list of terminal indices for each tree in each network.
    connections : list of list of list of functions
        A list of functions that define the connection between
        terminal points of the trees in each network. By default,
        the connection among n interpenetrating trees is defined
        by the midpoint (t=0.5) of spline curve that assigns the first
        two trees in the network.
    """
    assignments = []
    connections = []
    for i in range(forest.n_networks):
        network_assignments, network_connections = assign_network(forest, i, **kwargs)
        assignments.append(network_assignments)
        connections.append(network_connections)
    return assignments, connections


def geodesic(path_pts, start=None, end=None):
    ctrl_pts = numpy.vstack((start, path_pts, end))
    if path_pts.shape[0] > 3:
        k = 3
    elif path_pts.shape[0] > 2:
        k = 2
    else:
        k = 1
    tck = splprep(path_pts.T, s=0, k=k)
    geo_func = lambda t: numpy.array(splev(t, tck[0])).T
    return geo_func


def geodesic_cost(data, curve_generator=None, boundary_func=None, sample=20):
    curve = curve_generator(data.reshape(-1, 3))
    t = numpy.linspace(0, 1, sample)
    pts = curve(t)
    length = numpy.sum(numpy.linalg.norm(numpy.diff(pts, axis=0), axis=1))
    values = boundary_func(pts)
    values = numpy.exp(numpy.sum(values[values > 0]))
    cost = length * values
    return cost
