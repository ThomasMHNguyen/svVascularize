import numpy
import tqdm
from scipy.spatial import cKDTree
from collections import defaultdict
import pyvista as pv
import numpy as np

def extract_faces(surface, mesh, crease_angle: float = 60, verbose: bool = False, combine_walls: bool = True):
    """
    This function extracts the boundary domains from a given
    surface mesh file in PolyData format. By default, a
    separation angle is assumed to divide the C0 continuous
    manifold along disjoint C1 discontinuities which fail to
    meet the angle threshold criteria.

    Parameters
    ----------
        surface : PolyData surface mesh
        mesh : UnstructuredGrid
        crease_angle : float
        verbose : bool (default: False)
    Returns
    -------
        faces : dict
        iscap : list
        wall_surfaces: list of PolyData
        cap_surfaces: list of PolyData
    """
    if not surface.is_manifold:
        raise ValueError("Surface must be a righteous manifold for face extraction.")
    face_vertices = surface.faces.reshape(-1, 4)[:, 1:]
    unallocated_elements = set(range(face_vertices.shape[0]))
    vertex_map = build_vertex_map(face_vertices)
    edge_neighbors = build_edge_map(face_vertices, vertex_map)
    #face_neighbors = build_face_neighbors(surface)
    element_quality = surface.cell_quality(quality_measure='scaled_jacobian')
    # Use PyVista's auto-orientation with connectivity-based propagation
    # This correctly handles both outer surfaces and interior holes
    element_normals = surface.compute_normals(
        cell_normals=True,
        point_normals=False,
        auto_orient_normals=True,
        flip_normals=False,
        non_manifold_traversal=False
    ).cell_data["Normals"]
    collapsed_elements = numpy.isclose(element_quality.cell_data["scaled_jacobian"], 0.0, atol=1e-3)
    faces = partition(unallocated_elements, face_vertices, element_normals, crease_angle, vertex_map, edge_neighbors,
                      collapsed_elements)
    # Add a reconnecting step to anneal partitioned faces that are degenerate or collapsed into faces
    face_trees = []
    face_boundaries = []
    for face in faces:
        face_trees.append(cKDTree(surface.points[face_vertices[face, :].flatten(), :]))
        face_boundary = surface.extract_cells(face).extract_surface(algorithm='geometry').extract_feature_edges(boundary_edges=True,
                                                                                             manifold_edges=False,
                                                                                             feature_edges=False,
                                                                                             non_manifold_edges=False)
        face_boundaries.append([cKDTree(body.points) for body in face_boundary.split_bodies()])
    new_faces = []
    new_idx = []
    for i, face in enumerate(faces):
        face_cells = surface.extract_cells(face).extract_surface(algorithm='geometry')
        face_boundary = face_cells.extract_feature_edges(boundary_edges=True, manifold_edges=False,
                                                         feature_edges=False, non_manifold_edges=False)
        annealed = False
        if face_cells.n_points == face_boundary.n_points and len(face_boundary.split_bodies()) == 1:
            if verbose:
                print("Face {} is degenerate with no internal vertices.".format(i))
            for j, bounds in enumerate(face_boundaries):
                if i == j:
                    continue
                for k, bound in enumerate(bounds):
                    dists, _ = bound.query(face_boundary.points)
                    if numpy.all(numpy.isclose(dists, 0.0)):
                        if verbose:
                            print("Degenerate face {} annealed to face -> {}".format(i, j))
                        faces[j].extend(face)
                        annealed = True
                        break
                if annealed:
                    break
        #for j, tree in enumerate(face_trees):
        #    if i == j:
        #        continue
        #    dists, _ = tree.query(face_cells.points)
        #    if numpy.all(numpy.isclose(dists, 0.0)):
        #        if verbose:
        #            print("Degenerate face {} annealed to face -> {}".format(i, j))
        #        faces[j].extend(face)
        #        annealed = True
        #        break
        if not annealed:
            if verbose:
                print("Face {} is complete with internal vertices.".format(i))
            new_idx.append(i)
    for i in new_idx:
        new_faces.append(faces[i])
    faces = new_faces
    # Precompute boundary-loop KD-trees for all faces (used for robust matching)
    all_boundary_trees = []
    for i in range(len(faces)):
        f = surface.extract_cells(faces[i]).extract_surface(algorithm='geometry')
        loops = f.extract_feature_edges(boundary_edges=True,
                                        manifold_edges=False,
                                        feature_edges=False,
                                        non_manifold_edges=False)
        splits = loops.split_bodies()
        all_boundary_trees.append([cKDTree(splits[j].points) for j in range(splits.n_blocks)])
    iscap = []
    wall_faces = []
    cap_faces = []
    lumen_faces = []
    wall_boundaries = []
    wall_boundary_trees = []
    cap_boundaries = []
    cap_boundary_trees = []
    lumen_boundaries = []
    lumen_boundary_trees = []
    for i in range(len(faces)):
        f = surface.extract_cells(faces[i]).extract_surface(algorithm='geometry')
        tmp_bound_check = f.extract_feature_edges(boundary_edges=True, manifold_edges=False,
                                                 feature_edges=False, non_manifold_edges=False)
        tmp_bound_check = tmp_bound_check.split_bodies()
        n_boundary_loops = len(tmp_bound_check)

        # Multi-criteria classification using topology + geometry
        # Get normals for planarity check
        face_normals_array = f.compute_normals(cell_normals=True).cell_data["Normals"]
        mean_normal = numpy.mean(face_normals_array, axis=0)
        mean_normal = mean_normal / numpy.linalg.norm(mean_normal)

        # Check alignment of all normals with mean normal
        alignments = numpy.array([numpy.dot(n, mean_normal) for n in face_normals_array])
        planarity = numpy.mean(alignments)  # Close to 1.0 = flat, close to 0 = curved

        # Helper function to check if boundary-adjacent faces satisfy crease angle
        def check_boundary_crease_angles(surface_mesh, boundary_loops, crease_threshold):
            """
            Check if faces adjacent to boundary loops have creases.

            This checks if the surface has sharp discontinuities at the boundary,
            which would indicate it's a lumen that cleanly separates from surrounding surfaces.

            Returns True if boundary shows crease-like behavior (indicating a lumen),
            False otherwise (indicating smooth continuation).
            """
            if len(boundary_loops) < 2:
                return False  # Need at least 2 loops for lumen

            # Strategy: For each boundary edge, compute the expected normal direction
            # if the surface were to continue smoothly vs. the actual face normal.
            # A crease exists if there's a significant deviation.

            total_crease_edges = 0
            total_edges = 0

            for loop in boundary_loops:
                loop_points = loop.points
                if loop.n_points < 3:
                    continue

                # Compute loop centroid and approximate loop normal
                loop_centroid = numpy.mean(loop_points, axis=0)

                # Estimate loop plane normal using PCA or cross products
                # Simple approach: use first 3 points
                if loop.n_points >= 3:
                    v1 = loop_points[1] - loop_points[0]
                    v2 = loop_points[2] - loop_points[0]
                    loop_normal = numpy.cross(v1, v2)
                    loop_normal = loop_normal / (numpy.linalg.norm(loop_normal) + 1e-10)
                else:
                    continue

                # For each edge in the boundary loop
                for edge_idx in range(loop.n_points):
                    p1 = loop_points[edge_idx]
                    p2 = loop_points[(edge_idx + 1) % loop.n_points]
                    edge_midpoint = (p1 + p2) / 2.0

                    # Find the face containing this boundary edge
                    closest_cell_id = surface_mesh.find_closest_cell(edge_midpoint)
                    if closest_cell_id < 0:
                        continue

                    face_normal = face_normals_array[closest_cell_id]

                    # Check angle between face normal and loop normal
                    # For a lumen (cylinder), face normal should be roughly perpendicular to loop normal
                    # For a flat surface with hole, face normal should align with loop normal
                    dot_product = abs(numpy.dot(face_normal, loop_normal))
                    angle_to_loop = numpy.arccos(numpy.clip(dot_product, 0.0, 1.0)) * 180.0 / numpy.pi

                    total_edges += 1

                    # For a cylindrical lumen, face normals point radially outward,
                    # nearly perpendicular to the loop plane (angle close to 90°)
                    # For a flat surface, face normals align with loop normal (angle close to 0°)
                    # We want to identify lumens, so check if angle is far from 0°
                    if angle_to_loop > crease_threshold:
                        total_crease_edges += 1

            # If most boundary edges show large angles, it's likely a lumen
            if total_edges > 0:
                crease_ratio = total_crease_edges / total_edges
                return crease_ratio > 0.5  # More than 50% of edges show lumen-like geometry

            return False

        # Classification logic with multiple geometric criteria:
        # - Single boundary loop → CAP (simple endcap or inlet/outlet)
        # - Two or more boundary loops with boundary creases → LUMEN
        # - Two or more boundary loops without boundary creases → WALL

        is_cap = False
        is_lumen = False
        classification_reason = ""

        if n_boundary_loops == 1:
            # Single boundary loop: always a cap
            is_cap = True
            classification_reason = "single boundary loop"

        elif n_boundary_loops >= 2:
            # Two or more boundary loops: check if this is a lumen
            # NEW CRITERION: Boundary-adjacent faces must satisfy crease angle against boundary loop normals

            # First check: Do boundary-adjacent faces have creases?
            has_boundary_creases = check_boundary_crease_angles(f, tmp_bound_check, crease_angle)

            if not has_boundary_creases:
                # No creases at boundaries → this is a wall, not a lumen
                is_cap = False
                classification_reason = f"{n_boundary_loops} loops, no boundary creases (continuous surface)"
                if verbose:
                    print(f"Partition {i}: Detected as wall - {classification_reason}")
            else:
                # Has boundary creases → likely a lumen, verify with geometry
                # Use multiple geometric criteria as secondary checks

                if n_boundary_loops == 2:
                    # Criterion 1: Boundary loop circularity
                    # Circularity = 4π × Area / Perimeter² (1.0 = perfect circle, < 1.0 = irregular)
                    loop_0 = tmp_bound_check[0]
                    loop_1 = tmp_bound_check[1]

                    # Compute circularity for each loop
                    def compute_circularity(loop_polydata):
                        if loop_polydata.n_points < 3:
                            return 0.0
                        # Compute perimeter
                        perimeter = loop_polydata.length
                        # Estimate area using triangulation of loop
                        # Project to best-fit plane and compute 2D area
                        points = loop_polydata.points
                        centroid = numpy.mean(points, axis=0)
                        # Simple area estimate: sum of triangle areas from centroid
                        area = 0.0
                        for j in range(loop_polydata.n_points - 1):
                            p1 = points[j] - centroid
                            p2 = points[j + 1] - centroid
                            area += 0.5 * numpy.linalg.norm(numpy.cross(p1, p2))
                        # Add closing triangle
                        p1 = points[-1] - centroid
                        p2 = points[0] - centroid
                        area += 0.5 * numpy.linalg.norm(numpy.cross(p1, p2))

                        if perimeter > 0:
                            circularity = 4.0 * numpy.pi * area / (perimeter * perimeter)
                        else:
                            circularity = 0.0
                        return min(circularity, 1.0)  # Cap at 1.0 due to approximations

                    circularity_0 = compute_circularity(loop_0)
                    circularity_1 = compute_circularity(loop_1)

                    # Criterion 2: Boundary separation
                    centroid_0 = numpy.mean(loop_0.points, axis=0)
                    centroid_1 = numpy.mean(loop_1.points, axis=0)
                    separation = numpy.linalg.norm(centroid_1 - centroid_0)

                    # Estimate loop diameter (average of both loops)
                    diameter_0 = numpy.max(numpy.linalg.norm(loop_0.points - centroid_0, axis=1)) * 2.0
                    diameter_1 = numpy.max(numpy.linalg.norm(loop_1.points - centroid_1, axis=1)) * 2.0
                    avg_diameter = (diameter_0 + diameter_1) / 2.0

                    # Normalized separation (relative to diameter)
                    normalized_separation = separation / avg_diameter if avg_diameter > 0 else 0.0

                    # Decision logic: combine multiple criteria
                    is_lumen_candidate = False

                    # Strong indicator: both loops circular + separated
                    if circularity_0 > 0.7 and circularity_1 > 0.7 and normalized_separation > 0.5:
                        is_lumen_candidate = True
                        classification_reason = f"boundary creases + circular loops (C={circularity_0:.2f},{circularity_1:.2f}) + separated ({normalized_separation:.2f}×D)"

                    # Fallback: low planarity indicates curved surface
                    elif planarity < 0.5:
                        is_lumen_candidate = True
                        classification_reason = f"boundary creases + curved surface (planarity={planarity:.3f})"

                    # Additional check: high aspect ratio (long and thin)
                    elif normalized_separation > 2.0:
                        is_lumen_candidate = True
                        classification_reason = f"boundary creases + high aspect ratio (L/D={normalized_separation:.2f})"
                    else:
                        # Has creases but geometry doesn't match lumen → could be a wall with sharp edges
                        is_lumen_candidate = False
                        classification_reason = f"boundary creases but ambiguous geometry (C={circularity_0:.2f},{circularity_1:.2f}, sep={normalized_separation:.2f}×D)"

                    if is_lumen_candidate:
                        is_lumen = True
                        if verbose:
                            print(f"Partition {i}: Detected as lumen - {classification_reason}")
                    else:
                        if verbose:
                            print(f"Partition {i}: Detected as wall - {classification_reason}")
                else:
                    # 3+ loops with boundary creases: likely multi-outlet lumen
                    is_lumen = True
                    classification_reason = f"{n_boundary_loops} loops with boundary creases (multi-outlet lumen)"
                    if verbose:
                        print(f"Partition {i}: Detected as lumen - {classification_reason}")

        if is_lumen:
            # Lumen: surface with 2+ boundary loops and boundary creases
            tmp_lumen_boundaries = []
            tmp_lumen_boundary_trees = []
            for j in range(len(tmp_bound_check)):
                tmp_lumen_boundaries.append(tmp_bound_check[j])
                tmp_lumen_boundary_trees.append(cKDTree(tmp_bound_check[j].points))
            lumen_boundaries.append(tmp_lumen_boundaries)
            lumen_boundary_trees.append(tmp_lumen_boundary_trees)
            lumen_faces.append(faces[i])
            iscap.append(2)  # Mark as lumen in iscap array
        elif is_cap:
            # Cap: simple surface with single boundary loop
            iscap.append(1)
            cap_faces.append(faces[i])
            tmp_cap_boundaries = []
            tmp_cap_boundary_trees = []
            for j in range(len(tmp_bound_check)):
                tmp_cap_boundaries.append(tmp_bound_check[j])
                tmp_cap_boundary_trees.append(cKDTree(tmp_bound_check[j].points))
            cap_boundaries.append(tmp_cap_boundaries)
            cap_boundary_trees.append(tmp_cap_boundary_trees)
        else:
            # Wall: everything else
            tmp_wall_boundaries = []
            tmp_wall_boundary_trees = []
            boundaries = f.extract_feature_edges(boundary_edges=True, manifold_edges=False,
                                                 feature_edges=False, non_manifold_edges=False)
            boundaries = boundaries.split_bodies()
            for j in range(boundaries.n_blocks):
                tmp_wall_boundaries.append(boundaries[j])
                tmp_wall_boundary_trees.append(cKDTree(boundaries[j].points))
            wall_boundaries.append(tmp_wall_boundaries)
            wall_boundary_trees.append(tmp_wall_boundary_trees)
            iscap.append(0)
            wall_faces.append(faces[i])

    # Post-classification validation: a CAP must have a single boundary loop that
    # matches exactly one LUMEN boundary and no other face boundaries.
    # If a cap loop is shared among multiple faces or not shared with any lumen,
    # demote it to a wall.
    def boundaries_match(tree_a, tree_b, tol=1e-9):
        dists, _ = tree_a.query(tree_b.data)
        return numpy.all(numpy.isclose(dists, 0.0, atol=tol))

    for i in range(len(faces)):
        if iscap[i] != 1:
            continue
        # Require exactly one boundary loop on the cap face
        cap_loops = all_boundary_trees[i]
        if len(cap_loops) != 1:
            iscap[i] = 0
            continue
        cap_loop_tree = cap_loops[0]
        lumen_matches = 0
        other_matches = 0
        for j in range(len(faces)):
            if j == i:
                continue
            for other_loop in all_boundary_trees[j]:
                if boundaries_match(cap_loop_tree, other_loop):
                    if iscap[j] == 2:
                        lumen_matches += 1
                    else:
                        other_matches += 1
            # Early exit if already invalid
            if lumen_matches > 1 or other_matches > 0:
                break
        # Enforce: shared only with a single lumen loop
        if not (lumen_matches == 1 and other_matches == 0):
            iscap[i] = 0  # demote to wall
    # Rebuild type-specific face lists and boundary trees after cap validation
    wall_faces = []
    cap_faces = []
    lumen_faces = []
    wall_boundary_trees = []
    cap_boundary_trees = []
    lumen_boundary_trees = []
    for i in range(len(faces)):
        if iscap[i] == 0:
            wall_faces.append(faces[i])
            wall_boundary_trees.append(all_boundary_trees[i])
        elif iscap[i] == 1:
            cap_faces.append(faces[i])
            cap_boundary_trees.append(all_boundary_trees[i])
        elif iscap[i] == 2:
            lumen_faces.append(faces[i])
            lumen_boundary_trees.append(all_boundary_trees[i])

    # IMPORTANT: Caps and lumens should NEVER be merged with walls or each other
    # Only walls can be combined if they share boundaries. Prevent merges across
    # cap and lumen boundaries by providing them as non-wall constraints.
    all_non_wall_boundary_trees = cap_boundary_trees + lumen_boundary_trees

    if combine_walls and len(wall_faces) > 0:
        combined_wall_faces, combined_indices = combine_walls_graph_based(
            wall_faces, wall_boundary_trees, all_non_wall_boundary_trees, verbose=verbose
        )
    else:
        # No combination: each wall is its own component
        combined_wall_faces = wall_faces if len(wall_faces) > 0 else []
        combined_indices = [[i] for i in range(len(wall_faces))]

    # Build final face list: combined walls + individual caps + individual lumens (never merged)
    new_faces = []
    new_iscap = []
    for wall_face_list in combined_wall_faces:
        new_faces.append(wall_face_list)
        new_iscap.append(0)
    # Caps are kept separate - each cap is its own entity
    for i in range(len(cap_faces)):
        new_faces.append(cap_faces[i])
        new_iscap.append(1)
    # Lumens are kept separate - each lumen is its own entity
    for i in range(len(lumen_faces)):
        new_faces.append(lumen_faces[i])
        new_iscap.append(2)

    faces = new_faces
    iscap = new_iscap
    walls = []
    caps = []
    lumens = []
    for i, cap_code in enumerate(iscap):
        if cap_code == 0:
            walls.append(faces[i])
        elif cap_code == 1:
            caps.append(faces[i])
        elif cap_code == 2:
            lumens.append(faces[i])
    if not isinstance(mesh, type(None)):
        global_nodes = mesh.points
        global_node_tree = cKDTree(global_nodes)
        # Build a robust face->cell index using canonical keys (sorted tuples)
        face_to_cell = {}
        n_cells = mesh.n_cells
        for i in tqdm.trange(n_cells, desc="Indexing cell faces", leave=False):
            cell = mesh.GetCell(i)
            nfaces = cell.GetNumberOfFaces()
            for j in range(nfaces):
                face = cell.GetFace(j)
                npts = face.GetNumberOfPoints()
                # Canonicalize face nodes to a sorted tuple so orientation/order doesn't matter
                key = tuple(sorted(face.GetPointId(k) for k in range(npts)))
                #print("key: {} -> {}".format(key, i))
                face_to_cell[key] = i
    #for i, cap in enumerate(iscap):
    #    if not cap == 1:
    #        walls.append(faces[i])
    #    else:
    #        caps.append(faces[i])
    # Map the surface ids to the mesh ids for wall
    wall_surfaces = []
    wall_boundaries = []
    for i in tqdm.trange(len(walls), desc="Mapping wall surfaces <-> mesh ids", leave=False):
        wall_cells = surface.extract_cells(walls[i])
        wall_surface = wall_cells.extract_surface(algorithm='geometry')
        if not isinstance(mesh, type(None)):
            wall_surface.point_data["GlobalNodeID"] = numpy.zeros(wall_surface.n_points, dtype=numpy.int32)
            wall_surface.cell_data['GlobalElementID'] = numpy.zeros(wall_surface.n_cells, dtype=numpy.int32)
            wall_surface.cell_data['ModelFaceID'] = numpy.ones(wall_surface.n_cells, dtype=numpy.int32) * i
            _, indices = global_node_tree.query(wall_surface.points)
            wall_surface.point_data["GlobalNodeID"] = indices.astype(numpy.int32)
            # Assign Global Element IDs
            wall_faces = wall_surface.point_data["GlobalNodeID"][wall_surface.faces]
            wall_faces = wall_faces.reshape(-1, 4)[:, 1:].tolist()
            elem_ids = []
            for face in wall_faces:
                # Use the same canonical key used to build the map
                elem_ids.append(face_to_cell[tuple(sorted(face))])
            wall_surface.cell_data["GlobalElementID"] = numpy.array(elem_ids, dtype=numpy.int32)
            #wall_faces = numpy.sort(wall_faces, axis=1)
            #dists, indices = tet_face_tree.query(wall_faces)
            #if not numpy.all(numpy.isclose(dists, 0.0)):
            #    # Identify a small sample of mismatches for debugging
            #    bad_idx = numpy.where(~numpy.isclose(dists, 0.0))[0]
            #    sample = bad_idx[:5]
            #    examples = wall_faces[sample]
            #    raise ValueError(
            #        f"Failed to map all wall surface faces to volume mesh faces: {bad_idx.size} mismatches. "
            #        f"Example face node triples (GlobalNodeID) that failed exact match: {examples.tolist()}"
            #    )
            #wall_surface.cell_data["GlobalElementID"] = indices // 4
            #wall_surface.cell_data["GlobalElementID"] = wall_surface.cell_data["GlobalElementID"].astype(numpy.int32)

        boundaries = wall_surface.extract_feature_edges(boundary_edges=True, manifold_edges=False,
                                                             feature_edges=False, non_manifold_edges=False)
        boundaries = boundaries.split_bodies()
        tmp = []
        for j in range(boundaries.n_blocks):
            tmp.append(cKDTree(boundaries[j].points))
        wall_surfaces.append(wall_surface)
        wall_boundaries.append(tmp)
    # Map the surface ids to the mesh ids for the caps
    cap_surfaces = []
    cap_boundaries = []
    for i in tqdm.trange(len(caps), desc="Mapping cap surfaces <-> mesh ids", leave=False):
        face_cap = caps[i]
        cap_cells = surface.extract_cells(face_cap)
        cap_surface = cap_cells.extract_surface(algorithm='geometry')
        if not isinstance(mesh, type(None)):
            cap_surface.point_data["GlobalNodeID"] = numpy.zeros(cap_surface.n_points, dtype=numpy.int32)
            cap_surface.cell_data["GlobalElementID"] = numpy.zeros(cap_surface.n_cells, dtype=numpy.int32)
            cap_surface.cell_data["ModelFaceID"] = numpy.ones(cap_surface.n_cells, dtype=numpy.int32) * (len(walls) + i)
            # Assign Global Node IDs
            _, indices = global_node_tree.query(cap_surface.points)
            cap_surface.point_data["GlobalNodeID"] = indices.astype(numpy.int32)
            # Assign Global Element IDs
            cap_faces = cap_surface.point_data["GlobalNodeID"][cap_surface.faces]
            cap_faces = cap_faces.reshape(-1, 4)[:, 1:].tolist()
            elem_ids = []
            for face in cap_faces:
                elem_ids.append(face_to_cell[tuple(sorted(face))])
            cap_surface.cell_data["GlobalElementID"] = numpy.array(elem_ids, dtype=numpy.int32)
            #cap_faces = numpy.sort(cap_faces, axis=1)
            #dists, indices = tet_face_tree.query(cap_faces)
            #if not numpy.all(numpy.isclose(dists, 0.0)):
            #    bad_idx = numpy.where(~numpy.isclose(dists, 0.0))[0]
            #    sample = bad_idx[:5]
            #    examples = cap_faces[sample]
            #    raise ValueError(
            #        f"Failed to map all cap surface faces to volume mesh faces: {bad_idx.size} mismatches. "
            #        f"Example face node triples (GlobalNodeID) that failed exact match: {examples.tolist()}"
            #    )
            #cap_surface.cell_data["GlobalElementID"] = indices // 4
            #cap_surface.cell_data["GlobalElementID"] = cap_surface.cell_data["GlobalElementID"].astype(numpy.int32)
        boundaries = cap_surface.extract_feature_edges(boundary_edges=True, manifold_edges=False,
                                                           feature_edges=False, non_manifold_edges=False)
        boundaries = boundaries.split_bodies()
        tmp = []
        for j in range(boundaries.n_blocks):
            tmp.append(cKDTree(boundaries[j].points))
        cap_boundaries.append(tmp)
        cap_surfaces.append(cap_surface)

    # Map the surface ids to the mesh ids for the lumens
    lumen_surfaces = []
    lumen_boundaries = []
    for i in tqdm.trange(len(lumens), desc="Mapping lumen surfaces <-> mesh ids", leave=False):
        face_lumen = lumens[i]
        lumen_cells = surface.extract_cells(face_lumen)
        lumen_surface = lumen_cells.extract_surface(algorithm='geometry')
        if not isinstance(mesh, type(None)):
            lumen_surface.point_data["GlobalNodeID"] = numpy.zeros(lumen_surface.n_points, dtype=int)
            lumen_surface.cell_data["GlobalElementID"] = numpy.zeros(lumen_surface.n_cells, dtype=int)
            lumen_surface.cell_data["ModelFaceID"] = numpy.ones(lumen_surface.n_cells, dtype=int) * (len(walls) + len(caps) + i)
            # Assign Global Node IDs
            _, indices = global_node_tree.query(lumen_surface.points)
            lumen_surface.point_data["GlobalNodeID"] = indices.astype(numpy.int32)
            # Assign Global Element IDs
            lumen_faces = lumen_surface.point_data["GlobalNodeID"][lumen_surface.faces]
            lumen_faces = lumen_faces.reshape(-1, 4)[:, 1:].tolist()
            elem_ids = []
            for face in lumen_faces:
                elem_ids.append(face_to_cell[tuple(sorted(face))])
            lumen_surface.cell_data["GlobalElementID"] = numpy.array(elem_ids, dtype=numpy.int32)
            #lumen_faces = numpy.sort(lumen_faces, axis=1)
            #dists, indices = tet_face_tree.query(lumen_faces)
            #if not numpy.all(numpy.isclose(dists, 0.0)):
            #    bad_idx = numpy.where(~numpy.isclose(dists, 0.0))[0]
            #    sample = bad_idx[:5]
            #    examples = lumen_faces[sample]
            #    raise ValueError(
            #        f"Failed to map all lumen surface faces to volume mesh faces: {bad_idx.size} mismatches. "
            #        f"Example face node triples (GlobalNodeID) that failed exact match: {examples.tolist()}"
            #    )
            #lumen_surface.cell_data["GlobalElementID"] = indices // 4
            #lumen_surface.cell_data["GlobalElementID"] = lumen_surface.cell_data["GlobalElementID"].astype(numpy.int32)
        boundaries = lumen_surface.extract_feature_edges(boundary_edges=True, manifold_edges=False,
                                                           feature_edges=False, non_manifold_edges=False)
        boundaries = boundaries.split_bodies()
        tmp = []
        for j in range(boundaries.n_blocks):
            tmp.append(cKDTree(boundaries[j].points))
        lumen_boundaries.append(tmp)
        lumen_surfaces.append(lumen_surface)

    shared_boundaries = {}
    for i in tqdm.trange(len(cap_boundaries), desc="Mapping shared boundaries", leave=False):
        shared_boundaries[i] = []
        for j in range(len(cap_boundaries[i])):
            for k in range(len(wall_boundaries)):
                for l in range(len(wall_boundaries[k])):
                    dists, _ = cap_boundaries[i][j].query(wall_boundaries[k][l].data)
                    if numpy.all(numpy.isclose(dists, 0.0)):
                        shared_boundaries[i].append(k)
                        break
    return faces, wall_surfaces, cap_surfaces, lumen_surfaces, shared_boundaries

def split_polyline_by_turn_angle(polyline: pv.PolyData, angle_deg: float):
    """
    Split each polyline cell into multiple polyline segments whenever the
    turning angle at a vertex exceeds angle_deg.
    Returns a list[pv.PolyData], each a separate line/polyline.
    """
    out = []
    angle_thr = np.deg2rad(angle_deg)
    # Ensure we have only lines
    polyline = polyline.extract_cells(np.arange(polyline.n_cells)).clean()
    for ci in range(polyline.n_cells):
        cell = polyline.get_cell(ci)
        ids = np.array(cell.point_ids, dtype=int)
        pts = polyline.points[ids]
        # If the line is explicitly closed (last == first), remove the duplicate last point
        closed = len(ids) > 2 and ids[0] == ids[-1]
        if closed:
            ids = ids[:-1]
            pts = pts[:-1]
        n = len(ids)
        if n < 3:
            # Too short to split
            seg = pv.PolyData(polyline.points[ids])
            seg.lines = np.hstack([[n], np.arange(n)]).astype(np.int64)
            out.append(seg)
            continue
        # Compute turning angles at interior vertices
        # angle at i uses vectors (i - i-1) and (i+1 - i)
        break_idx = []
        for i in range(1, n - 1):
            v0 = pts[i] - pts[i - 1]
            v1 = pts[i + 1] - pts[i]
            n0 = np.linalg.norm(v0)
            n1 = np.linalg.norm(v1)
            if n0 == 0 or n1 == 0:
                continue
            c = np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0)
            ang = np.arccos(c)
            if ang > angle_thr:
                break_idx.append(i)
        # Build segments [start .. break] [break .. nextbreak] ... [last]
        cuts = [0] + break_idx + [n - 1]
        for a, b in zip(cuts[:-1], cuts[1:]):
            # include b in segment
            seg_ids = ids[a:b + 1]
            if len(seg_ids) < 2:
                continue
            seg_pts = polyline.points[seg_ids]
            seg = pv.PolyData(seg_pts)
            seg.lines = np.hstack([[len(seg_pts)], np.arange(len(seg_pts))]).astype(np.int64)
            out.append(seg)
        # If it’s a closed loop, you might also want to test/split the wrap-around corner.
        # For rectangles, the above is usually enough once the polyline ordering is good.
    return [s.clean() for s in out]


def has_matching_boundary(boundary_trees_a, boundary_trees_b, tolerance=1e-9):
    """
    Check if two walls share at least one complete matching boundary.

    Parameters
    ----------
    boundary_trees_a : list of cKDTree
        Boundary trees for first wall
    boundary_trees_b : list of cKDTree
        Boundary trees for second wall
    tolerance : float
        Distance tolerance for point matching

    Returns
    -------
    bool
        True if walls share a complete matching boundary
    """
    for tree_a in boundary_trees_a:
        for tree_b in boundary_trees_b:
            dists, _ = tree_a.query(tree_b.data)
            if numpy.all(numpy.isclose(dists, 0.0, atol=tolerance)):
                return True
    return False


def check_all_boundaries_matched(wall_idx, wall_boundary_trees, cap_boundary_trees, tolerance=1e-9):
    """
    Check if all boundaries of a wall are either matched to other walls/caps or are complete.

    Parameters
    ----------
    wall_idx : int
        Index of wall to check
    wall_boundary_trees : list of list of cKDTree
        Boundary trees for all walls
    cap_boundary_trees : list of list of cKDTree
        Boundary trees for all caps
    tolerance : float
        Distance tolerance for point matching

    Returns
    -------
    bool
        True if all boundaries are properly matched
    """
    for m in range(len(wall_boundary_trees[wall_idx])):
        boundary_matched = False

        # Check against other walls
        for ii in range(len(wall_boundary_trees)):
            if ii == wall_idx:
                continue
            for kk in range(len(wall_boundary_trees[ii])):
                dists, _ = wall_boundary_trees[wall_idx][m].query(wall_boundary_trees[ii][kk].data)
                if numpy.all(numpy.isclose(dists, 0.0, atol=tolerance)):
                    boundary_matched = True
                    break
            if boundary_matched:
                break

        # Check against caps
        if not boundary_matched:
            for ii in range(len(cap_boundary_trees)):
                for kk in range(len(cap_boundary_trees[ii])):
                    dists, _ = wall_boundary_trees[wall_idx][m].query(cap_boundary_trees[ii][kk].data)
                    if numpy.all(numpy.isclose(dists, 0.0, atol=tolerance)):
                        boundary_matched = True
                        break
                if boundary_matched:
                    break

        # If any boundary is partially matched (split), reject
        if not boundary_matched:
            # Check for partial matches (boundary is split)
            for ii in range(len(wall_boundary_trees)):
                if ii == wall_idx:
                    continue
                for kk in range(len(wall_boundary_trees[ii])):
                    dists, _ = wall_boundary_trees[wall_idx][m].query(wall_boundary_trees[ii][kk].data)
                    if numpy.any(numpy.isclose(dists, 0.0, atol=tolerance)) and not numpy.all(numpy.isclose(dists, 0.0, atol=tolerance)):
                        return False

    return True


def combine_walls_graph_based(wall_faces, wall_boundary_trees, cap_boundary_trees, verbose=False):
    """
    Combine walls using a graph-based approach with connected components.

    Parameters
    ----------
    wall_faces : list of list
        Face indices for each wall
    wall_boundary_trees : list of list of cKDTree
        Boundary trees for each wall
    cap_boundary_trees : list of list of cKDTree
        Boundary trees for each cap
    verbose : bool
        Print debug information

    Returns
    -------
    tuple
        (combined_wall_faces, combined_indices) where combined_wall_faces is a list
        of combined face lists and combined_indices tracks which walls were combined
    """
    n_walls = len(wall_faces)
    if n_walls == 0:
        return [], []

    # Build adjacency matrix for walls that share boundaries
    adjacency = numpy.zeros((n_walls, n_walls), dtype=bool)

    for i in range(n_walls):
        for j in range(i + 1, n_walls):
            if has_matching_boundary(wall_boundary_trees[i], wall_boundary_trees[j]):
                # Check if both walls have all boundaries properly matched
                if check_all_boundaries_matched(i, wall_boundary_trees, cap_boundary_trees) and \
                   check_all_boundaries_matched(j, wall_boundary_trees, cap_boundary_trees):
                    adjacency[i, j] = adjacency[j, i] = True

    # Find connected components using depth-first search
    visited = numpy.zeros(n_walls, dtype=bool)
    combined_wall_faces = []
    combined_indices = []

    for i in range(n_walls):
        if not visited[i]:
            # Start new component
            component_faces = []
            component_indices = []
            stack = [i]

            while stack:
                node = stack.pop()
                if visited[node]:
                    continue

                visited[node] = True
                component_faces.extend(wall_faces[node])
                component_indices.append(node)

                # Add unvisited neighbors to stack
                neighbors = numpy.where(adjacency[node])[0]
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        stack.append(neighbor)

            combined_wall_faces.append(component_faces)
            combined_indices.append(component_indices)

            if verbose and len(component_indices) > 1:
                print(f"Combined walls {component_indices} into single surface")

    return combined_wall_faces, combined_indices


def build_vertex_map(face_vertices):
    n_points = int(numpy.max(face_vertices) + 1)
    vertex_map = [[] for i in range(n_points)]
    for i in tqdm.trange(face_vertices.shape[0], desc="Building vertex map", leave=False):
        for j in range(face_vertices.shape[1]):
            vertex_map[int(face_vertices[i, j])].append(i)
    for i in range(len(vertex_map)):
        vertex_map[i] = tuple(vertex_map[i])
    vertex_map = tuple(vertex_map)
    return vertex_map


def build_edge_map(face_vertices, vertex_map):
    neighbors = {}
    for i in tqdm.trange(face_vertices.shape[0], desc="Building edge map", leave=False):
        face = face_vertices[i, :]
        num_points = len(face)
        tmp = []
        for j in range(num_points):
            vrt_1 = set(vertex_map[face[j]])
            vrt_2 = set(vertex_map[face[(j + 1) % num_points]])
            vrt = vrt_1.intersection(vrt_2)
            vrt.remove(i)
            for k in vrt:
                tmp.append(k)
        neighbors[i] = tuple(tmp)
    return neighbors


def build_face_map(surface):
    face_map = {}
    for i in tqdm.trange(surface.n_cells, desc="Building face map", leave=False):
        neighbors = surface.cell_neighbors(i, connections='edges')
        face_map[i] = tuple(neighbors)
    return face_map


def build_edge_map_2(surface):
    edge_map = {}
    for i in tqdm.trange(surface.n_cells, desc="Building edge map", leave=False):
        cell = surface.get_cell(i)
        for j in range(cell.n_edges):
            edge = tuple(sorted(cell.edges[j].point_ids))
            if edge not in edge_map.keys():
                edge_map[edge] = [i]
            else:
                edge_map[edge].append(i)
    face_map = {}
    for i in range(surface.n_cells):
        neighbors = []
        cell = surface.get_cell(i)
        for j in range(cell.n_edges):
            edge = tuple(sorted(cell.edges[j].point_ids))
            neighbors.extend(edge_map[edge])
        neighbors = list(set(neighbors))
        neighbors.remove(i)
        face_map[i] = tuple(neighbors)
    return face_map


def build_face_neighbors(mesh):
    """
    Build a list of face neighbors for each face in a PyVista PolyData surface mesh.
    Parameters:
    - mesh (pv.PolyData): Input PyVista PolyData surface mesh.
    Returns:
    - neighbors_list (list of lists): For each face, a list of indices of neighboring faces.
    """
    # Ensure we're working with a clean triangulated mesh
    edges_to_cells = defaultdict(set)
    neighbors_list = [[] for _ in range(mesh.n_cells)]
    # Build the mapping from edges to cells
    for i in range(mesh.n_cells):
        cell_point_ids = mesh.get_cell(i).point_ids
        edges = [(cell_point_ids[j], cell_point_ids[(j + 1) % 3]) for j in range(3)]
        for edge in edges:
            sorted_edge = tuple(sorted(edge))  # Sort the tuple to avoid directional issues
            edges_to_cells[sorted_edge].add(i)
    # Use the mapping to find neighbors
    for edge, cells in edges_to_cells.items():
        for cell_id in cells:
            neighbors_list[cell_id].extend(cells - {cell_id})
    # Remove duplicates and sort (optional, for cleaner output)
    neighbors_list = [sorted(list(set(neighbors))) for neighbors in neighbors_list]
    neighbors_map = {}
    for i in range(len(neighbors_list)):
        neighbors_map[i] = tuple(neighbors_list[i])
    return neighbors_map


def correct_normals(cell_normals, face_vertices, mesh):
    """
    Correct normal orientation using geometric center-based approach.
    This is more robust than using point normals at crease boundaries.
    """
    # Compute face centers
    face_centers = numpy.zeros((face_vertices.shape[0], 3), dtype=float)
    for i in range(face_vertices.shape[0]):
        face_centers[i] = numpy.mean(mesh.points[face_vertices[i], :], axis=0)

    # Compute mesh centroid
    mesh_center = numpy.mean(mesh.points, axis=0)

    # Orient normals: should point away from centroid for convex objects
    # For surfaces, we want consistent orientation
    for i in tqdm.trange(cell_normals.shape[0], desc="Correcting cell normals", leave=False):
        to_center = mesh_center - face_centers[i]
        # If normal points toward center, flip it
        if numpy.dot(to_center, cell_normals[i, :]) > 0:
            cell_normals[i, :] *= -1

    return cell_normals


def partition(unallocated, face_vertices, cell_normals, angle, vertex_map, edge_neighbors, collapsed):
    faces = []
    allocated = set()
    collapse_check = {}
    #os.mkdir('partition_algorithm')
    while len(unallocated) > 0:
        current_element = unallocated.pop()
        #if collapsed[current_element]:
        #    unallocated.update([current_element])
        #    continue
        allocated.update([current_element])
        check = set(tuple([current_element]))
        current_face = []

        while len(check) > 0:
            current_id = check.pop()
            current_normal = cell_normals[current_id, :]

            neighbors = edge_neighbors[current_id]
            neighbors = set(neighbors)
            neighbors = neighbors.difference(allocated)
            neighbors = neighbors.difference(check)

            for i in neighbors:
                # Compare current element against its immediate neighbor
                face_angle = numpy.arccos(numpy.clip(numpy.dot(current_normal, cell_normals[i, :]), -1, 1))
                face_angle = (face_angle/numpy.pi)*180.0
                if face_angle < angle:
                    check.update([i])
                #elif face_angle > angle and collapsed[i]:
                #    unallocated.remove(i)
                #    allocated.update([i])
                #    current_face.append(i)
            if current_id in unallocated:
                unallocated.remove(current_id)
            allocated.update([current_id])
            current_face.append(current_id)
            #if len(faces) == 0:
            #    plotter = pyvista.Plotter()
            #    plotter.add_mesh(mesh, color='grey', opacity=0.15)
            #    accepted_cells = mesh.extract_cells(current_face)
            #    check_cells = mesh.extract_cells(list(check))
            #    neighbor_cells = mesh.extract_cells(list(neighbors))
            #    plotter.add_mesh(accepted_cells, color='green')
            #    plotter.add_mesh(check_cells, color='yellow')
            #    plotter.add_mesh(neighbor_cells, color='red')
            #    plotter.show()
        faces.append(current_face)
    return faces
