import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import h5py

# Coordinates range from 0 to 150
MAX_COORD = 150

def particle_bounds(coordinates, x_bounds, y_bounds):
    x_mid = np.mean(x_bounds)
    y_mid = np.mean(y_bounds)
    acc = 0
    if coordinates[0] >= x_mid:
        acc += 2
    if coordinates[1] >= y_mid:
        acc += 1
    return acc

class BHQuadNode:
    def __init__(self, coordinates, masses, x_bounds, y_bounds,
                 parent=None, max_leaf=12):
        self.N = masses.size
        self.parent = parent
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

        if self.N > max_leaf:
            # Need to split
            x_mid = np.mean(x_bounds)
            y_mid = np.mean(y_bounds)

            # Select all the particles according to the new bound
            left_of_mid = coordinates[:, 0] < x_mid
            right_of_mid = np.logical_not(left_of_mid)
            below_mid = coordinates[:, 1] < y_mid
            above_mid = np.logical_not(below_mid)

            ll_selection = np.logical_and(left_of_mid, below_mid)
            lh_selection = np.logical_and(left_of_mid, above_mid)
            rl_selection = np.logical_and(right_of_mid, below_mid)
            rh_selection = np.logical_and(right_of_mid, above_mid)

            # Construct the new nodes
            ll_node = BHQuadNode(coordinates[ll_selection], masses[ll_selection],
                                 (x_bounds[0], x_mid), (y_bounds[0], y_mid),
                                 self, max_leaf)
            lh_node = BHQuadNode(coordinates[lh_selection], masses[lh_selection],
                                 (x_bounds[0], x_mid), (y_mid, y_bounds[1]),
                                 self, max_leaf)
            rl_node = BHQuadNode(coordinates[rl_selection], masses[rl_selection],
                                 (x_mid, x_bounds[1]), (y_bounds[0], y_mid),
                                 self, max_leaf)
            rh_node = BHQuadNode(coordinates[rh_selection], masses[rh_selection],
                                 (x_mid, x_bounds[1]), (y_mid, y_bounds[1]),
                                 self, max_leaf)
            self.nodes = (ll_node, lh_node, rl_node, rh_node)

            # Calculate masses and CoM using result in nodes
            node_masses = np.array([node.mass for node in self.nodes])
            node_CoMs = np.array([node.CoM for node in self.nodes])
            self.mass = np.sum(node_masses)
            self.CoM = np.average(node_CoMs, axis=0, weights=node_masses)
        else:
            # We are a leaf node
            self.mass = np.sum(masses)
            self.nodes = None
            if self.mass == 0:
                self.CoM = np.array((np.mean(x_bounds), np.mean(y_bounds)))
            else:
                self.CoM = np.average(coordinates, axis=0, weights=masses)


with h5py.File('DataFiles/colliding.hdf5') as f:
    coordinates = np.array(f['PartType4/Coordinates'])[:, :2]
    masses = np.array(f['PartType4/Masses'])

tree = BHQuadNode(coordinates, masses, (0, MAX_COORD), (0, MAX_COORD))

fig, ax = plt.subplots(1)

ax.scatter(coordinates[:, 0], coordinates[:, 1], s=1)
ax.set_xlim(0, MAX_COORD)
ax.set_ylim(0, MAX_COORD)
ax.set_aspect('equal')

ax.set_xlabel('x')
ax.set_ylabel('y')

# Pre-order (top-down) traversal of the tree,
# using a python list as stack
stack = [tree]
# While we still have nodes to print
while stack:
    # Print the current node and add children to the stack
    current_node = stack.pop()
    x_bounds = current_node.x_bounds
    y_bounds = current_node.y_bounds
    width = x_bounds[1] - x_bounds[0]
    height = y_bounds[1] - y_bounds[0]
    rect = Rectangle((x_bounds[0], y_bounds[0]), width, height, fill=False)
    ax.add_patch(rect)
    if current_node.nodes:
        [stack.append(node) for node in current_node.nodes]
plt.savefig('plots/ex7_whole_tree.pdf')
plt.close()

# First find the particle in the tree
search_route = [tree]
# I.e. while the last visited node is not a leaf
while search_route[-1].nodes:
    # Find the child that contains the particle
    x_bounds = search_route[-1].x_bounds
    y_bounds = search_route[-1].y_bounds
    index = particle_bounds(coordinates[100], x_bounds, y_bounds)
    search_route.append(search_route[-1].nodes[index])

print('Found particle with index 100')
# Search finished, walk back up the tree
while search_route:
    current_node = search_route.pop()
    print(f'Node mass: {current_node.mass:.3f}')
