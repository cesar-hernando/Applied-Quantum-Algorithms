'''
In this paper, we perform the fourier feature map described in Ref 3, which is used as input for LASSO regression
'''

import numpy as np
import functions

def get_local_edges(input_edge, dim_grid, delta=1):
    """
    Return all lattice edges within L1 distance `delta` from the midpoint of `input_edge`,
    separated into horizontal and vertical edges, organized as 2D arrays.
    """
    rows, cols = dim_grid

    def edge_midpoint(edge):
        (x1, y1), (x2, y2) = edge
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def l1_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    # Initialize edge matrices
    horizontal_edges = np.empty((rows, cols - 1), dtype=object)
    vertical_edges = np.empty((rows - 1, cols), dtype=object)

    # Fill edge matrices and collect midpoints
    for x in range(rows):
        for y in range(cols):
            if y + 1 < cols:
                horizontal_edges[x, y] = ((x, y), (x, y + 1))
            if x + 1 < rows:
                vertical_edges[x, y] = ((x, y), (x + 1, y))

    # Get midpoint of the input edge
    center = edge_midpoint(input_edge)

    # Compute local masks
    local_horizontal = np.zeros((rows, cols - 1), dtype=bool)
    for x in range(rows):
        for y in range(cols - 1):
            if horizontal_edges[x, y] is not None:
                mid = edge_midpoint(horizontal_edges[x, y])
                local_horizontal[x, y] = l1_distance(mid, center) <= delta

    local_vertical = np.zeros((rows - 1, cols), dtype=bool)
    for x in range(rows - 1):
        for y in range(cols):
            if vertical_edges[x, y] is not None:
                mid = edge_midpoint(vertical_edges[x, y])
                local_vertical[x, y] = l1_distance(mid, center) <= delta

    return local_horizontal, local_vertical


def generate_local_couplings(J_right, J_down, local_h, local_v):
    J_right_local = J_right[local_h]
    J_down_local = J_down[local_v]
    z = np.concatenate((J_right_local, J_down_local))

    return z

def random_fourier_feature_map(Z, w, gamma, R):
    l = len(Z)
    p = gamma/np.sqrt(l)
    phi = np.zeros(l*2*R)
    for i, z in enumerate(Z):
        for j in range(2*R):
            if j % 2 == 0:
                phi[i*2*R+j] = np.cos(p*w[j//2]*z)
            else:
                phi[i*2*R+j] = np.sin(p*w[j//2]*z)

    return phi
    




