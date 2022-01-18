# Lib import
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate
from shapely.geometry import Polygon
from typing import Dict, Tuple, List

# Our files
from heuristic import calculate_heuristics
from definitions import Color, Edge, Node, Position

class Graph:
    def __init__(self, img):
        """
        Primary constructor. Builds a fully connected neighborhood graph and stores the base image provided.

        :param
            img: np.ndarray, the template image to store and from which to build the graph.
        """
        self.img = img
        self.nodes: Dict[Position, Node] = {}
        self.splines: List[Tuple[List[Position], Color]] = []

        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                pixel_color = img[y, x]
                new_node = Node(x, y, pixel_color)

                window = np.linspace(-1, 1, 3).astype(int)

                for i in range(len(window)):
                    for j in range(len(window)):
                        new_node.neighbors.append((x + window[i], y + window[j]))

                self.nodes[(x, y)] = new_node

        self.clear_nodes_neighbors()

    ##### Class Utilities #####

    def RGB_to_YUV(self):
        """
        Converts the underlying image color values from RGB to YUV. Also updates the individual pixels color values.
        """
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2YUV)
        for y in range(self.img.shape[0]):
            for x in range(self.img.shape[1]):
                self.nodes[(x, y)].color = self.img[y, x]


    def YUV_to_RGB(self):
        """
        Converts the underlying image color values from YUV to RGB. Also updates the individual pixels color values.
        """
        self.img = cv2.cvtColor(self.img, cv2.COLOR_YUV2RGB)
        for y in range(self.img.shape[0]):
            for x in range(self.img.shape[1]):
                self.nodes[(x, y)].color = self.img[y, x]


    def is_neighbor(self, node: Node, neighbor: Position):
        """
        Returns true if neighbor is in the neighbors of the node.

        :param
            node: Node, a node from which to check the neighbors
            neighbor: Position, the node position to check for
        """
        if node not in self.nodes.keys():
            return False
        return neighbor in self.nodes[node].neighbors


    def is_on_img_border(self, position: Position):
        """
        Returns true if the given position is situated in the borders of the image. 
        """
        return position[0] in [0, self.img.shape[1]] or position[1] in [0, self.img.shape[0]]


    def get_darker_node(self, node1, node2):
        """
        Returns the node with the darkest (lowest) Y color value. Color format is assumed to be YUV (see `RGB_to_YUV`).
        """
        if node1.color[0] > node2.color[0]: return node2
        return node1
        

    def find_edge_from_corner(self, position: Position, color: Color, activeEdges: List[Tuple[Edge, Node]]):
        """
        Extracts the edge containing the given position, with the given color value.

        :param
            position: Position, the position from which to check the edges.
            color: Color, the color value to match.
            activeEdges: List[Tuple[Edge, Node]], the list of active edges as defined in the paper, and as built in `extract_splines`.
        
        :return
            edge: Edge, the corresponding edge, if found, otherwise None.
        """
        result = [edge for edge in activeEdges if position in [edge[0][0], edge[0][1]] and np.array_equal(edge[1].color, color)]
        if len(result) != 1:
            return None
        return result[0]


    def clear_nodes_neighbors(self):
        """
        Remove non possible neighbors based on the image shape (i.e. neighbors to positions outside the image).
        """
        for node in self.nodes.values():
            new_neighbors = []
            for x, y in node.neighbors:
                if x < 0 or y < 0 or x > self.img.shape[1] - 1 or y > self.img.shape[0] - 1:
                    continue
                if (x, y) == node.position:
                    continue

                new_neighbors.append((x, y))
            node.neighbors = new_neighbors
    
    ##### Pipeline Functions #####

    def keep_YUV_connection(self):
        """
        Eliminates graph connections from pixels that have sufficiently different YUV color values.
        """
        for node in self.nodes.values():
            new_neighbors = []
            for n in node.neighbors:
                n_color = self.img[n[1], n[0]]

                dY = abs(int(node.color[0]) - int(n_color[0]))
                dU = abs(int(node.color[1]) - int(n_color[1]))
                dV = abs(int(node.color[2]) - int(n_color[2]))

                if dY > 48 or dU > 7 or dV > 6:
                    continue

                new_neighbors.append(n)

            node.neighbors = new_neighbors


    def remove_duplicate_connection(self):
        """
        Removes redundant connections in the neighborhood graph: if a diagonal is detected and the 
        connection between the two connected nodes is already existing through cartesian connections,
        then the diagonal is removed.
        """
        for node in self.nodes.values():
            nx = node.position[0]
            ny = node.position[1]
            new_neighbors = []
            for x, y in node.neighbors:
                if (x, y) == (nx+1, ny+1) and (nx+1, ny) in node.neighbors and (nx, ny+1) in node.neighbors:
                    continue
                if (x, y) == (nx-1, ny-1) and (nx-1, ny) in node.neighbors and (nx, ny-1) in node.neighbors:
                    continue
                if (x, y) == (nx-1, ny+1) and (nx-1, ny) in node.neighbors and (nx, ny+1) in node.neighbors:
                    continue
                if (x, y) == (nx+1, ny-1) and (nx+1, ny) in node.neighbors and (nx, ny-1) in node.neighbors:
                    continue

                new_neighbors.append((x, y))

            node.neighbors = new_neighbors


    def apply_heuristics(self):
        """
        Applies the heuristic functions of the heuristic file, weighting each problematic pairs of edges,
        and cutting the one with the lowest resulting weight.
        """
        for x in range(self.img.shape[1] - 1):
            for y in range(self.img.shape[0] - 1):
                top_left = self.nodes[(x, y)]
                top_right = self.nodes[(x + 1, y)]
                bottom_left = self.nodes[(x, y + 1)]
                bottom_right = self.nodes[(x + 1, y + 1)]

                if bottom_right.position not in top_left.neighbors:
                    continue
                if bottom_left.position not in top_right.neighbors:
                    continue

                # Case of crossing edges
                (weight1, weight2) = calculate_heuristics(self.nodes, (x, y))

                if weight1 <= weight2:
                    # Cut weight1
                    self.nodes[(x, y)].neighbors.remove((x+1, y+1))  # top left remove bottom right
                    self.nodes[(x+1, y+1)].neighbors.remove((x, y))  # bottom right remove top left

                if weight1 >= weight2:
                    # Cut weight2
                    self.nodes[(x, y+1)].neighbors.remove((x+1, y))  # bottom left remove top right
                    self.nodes[(x+1, y)].neighbors.remove((x, y+1))  # top right remove bottom left


    def generate_voronoi_corners(self):
        """
        Creates pixel corners for each image pixel, and reshapes them in a Voronoi diagram,
        with the constraint that each node must share an edge with the edges it's connected with in the similarity graph.

        The corner list follow a particular order starting from the top left corner and continuing in anticlockwise.
        Exemple of the corners of a pixel :
            TOP LEFT      TOP         TOP RIGHT
            LEFT                      RIGHT
            BOTTOM LEFT   BOTTOM      BOTTOM RIGHT
        """
        # Add the new coordinates of the corners of all pixels.
        # We check the connections between the diagonal pixels that surrounds us to know if we are expanding the area.
        for node in self.nodes.values():
            center = (node.position[0] + 0.5, node.position[1] + 0.5)
            position = node.position

            # TOP LEFT CORNER
            # Check if the current pixel has a connection with his TOP LEFT neighbor
            if (position[0] - 1, position[1] - 1) in node.neighbors:
                node.corners.append((center[0] - 0.25, center[1] - 0.75))
                node.corners.append((center[0] - 0.75, center[1] - 0.25))
            # Check if the TOP (x, y-1) pixel has a connection with his BOTTOM LEFT (x-1, y+1) pixel
            elif self.is_neighbor((position[0], position[1] - 1), (position[0]-1, position[1])):
                node.corners.append((center[0] - 0.25, center[1] - 0.25))
            else:
                node.corners.append((center[0] - 0.5, center[1] - 0.5))

            # LEFT CORNER
            node.corners.append((center[0] - 0.5, center[1]))

            # BOTTOM LEFT CORNER
            # Check if the current pixel has a connection with his BOTTOM LEFT neighbor
            if (position[0] - 1, position[1] + 1) in node.neighbors:
                node.corners.append((center[0] - 0.75, center[1] + 0.25))
                node.corners.append((center[0] - 0.25, center[1] + 0.75))
            # Check if the BOTTOM (x, y+1) pixel has a connection with his TOP LEFT (x-1, y-1) pixel
            elif self.is_neighbor((position[0], position[1]+1), (position[0]-1, position[1])):
                node.corners.append((center[0] - 0.25, center[1] + 0.25))
            else:
                node.corners.append((center[0] - 0.5, center[1] + 0.5))

            # BOTTOM CORNER
            node.corners.append((center[0], center[1] + 0.5))

            # BOTTOM RIGHT CORNER
            # Check if the current pixel has a connection with his BOTTOM RIGHT neighbor
            if (position[0] + 1, position[1] + 1) in node.neighbors:
                node.corners.append((center[0] + 0.25, center[1] + 0.75))
                node.corners.append((center[0] + 0.75, center[1] + 0.25))
            # Check if the BOTTOM (x, y+1) pixel has a connection with his TOP RIGHT (x+1, y-1) pixel
            elif self.is_neighbor((position[0], position[1]+1), (position[0]+1, position[1])):
                node.corners.append((center[0] + 0.25, center[1] + 0.25))
            else:
                node.corners.append((center[0] + 0.5, center[1] + 0.5))

            # RIGHT CORNER
            node.corners.append((center[0] + 0.5, center[1]))

            # TOP RIGHT CORNER
            # Check if the current pixel has a connection with his TOP RIGHT neighbor
            if (position[0] + 1, position[1] - 1) in node.neighbors:
                node.corners.append((center[0] + 0.75, center[1] - 0.25))
                node.corners.append((center[0] + 0.25, center[1] - 0.75))
            # Check if the TOP (x, y-1) pixel has a connection with his BOTTOM RIGHT (x+1, y+1) pixel
            elif self.is_neighbor((position[0], position[1] - 1), (position[0]+1, position[1])):  #  
                node.corners.append((center[0] + 0.25, center[1] - 0.25))
            else:
                node.corners.append((center[0] + 0.5, center[1] - 0.5))

            # TOP CORNER
            node.corners.append((center[0], center[1] - 0.5))


    def collapse_valence2(self):
        """
        Simplifies the voronoi diagram built with `generate_voronoi_diagram` as the paper suggests:
        for each pixel corner, if it is connected with only 2 other corner, remove this corner entirely,
        and connect it's neighbors instead.
        """
        valencies: Dict[Position, int] = {}     # Stock the valency value of all corners
        for node in self.nodes.values():
            for i in range(len(node.corners)):
                left_corner = node.corners[i]
                right_corner = node.corners[(i+1) % len(node.corners)]  # A region is closed by the corners
                if left_corner not in valencies:
                    valencies[left_corner] = 1
                else:
                    valencies[left_corner] += 1
                if right_corner not in valencies:
                    valencies[right_corner] = 1
                else:
                    valencies[right_corner] += 1

        for node in self.nodes.values():
            tmp = []
            for corner in node.corners:
                # Keep only the corners with more than 2 neighbors or are on the image border (to avoid to smooth the image corners)
                if self.is_on_img_border(corner) or valencies[corner] != 4:     
                    tmp.append(corner)
            node.corners = tmp


    def extract_splines(self):
        """
        Build the internal list of splines form the voronoi diagram obtained with `collapse_valence2`,
        using visible edges, and using corners positions as control points.
        To display the result of these splines, see `print_splines`.
        """
        edges: Dict[Edge, Node] = {}
        activeEdges: List[Tuple[Edge, Node]] = []

        # This loop finds all "active edges" as defined in the paper in a list (defined above).
        for node in self.nodes.values():
            for i in range(len(node.corners)):
                left_corner = node.corners[i]
                # The region is closed by the corners, so we use the modulo to consider the edge between the last and first corner.
                right_corner = node.corners[(i+1) % len(node.corners)]
                if (right_corner, left_corner) in edges:
                    n = edges[(right_corner, left_corner)]
                    if n.position == node.position:
                        continue
                    if np.array_equal(n.color, node.color):
                        continue
                    
                    # If we find the same edge twice, we add it as an "active edge", and store the darkest color.
                    activeEdges.append(((left_corner, right_corner), self.get_darker_node(n, node)))  # Edge, Node

                else:
                    edges[(left_corner, right_corner)] = node

        # This loop extracts control points for each splines from the active edges.
        # NOTE: This block doesn't quite work, as the extracted splines can contain some inconsistencies like non-closing loop or unusual control point.
        while len(activeEdges) != 0:
            initialEdge = activeEdges[0]
            newSpline: Tuple[List[Position], Color] = ([initialEdge[0][0], initialEdge[0][1]], initialEdge[1].color)

            currentCorner = initialEdge[0][0]
            activeEdges.remove(initialEdge)
            currentEdge = self.find_edge_from_corner(currentCorner, newSpline[1], activeEdges)
            shouldConnect = True

            # We explore a spline one way completely.
            while currentEdge != None:
                currentCorner = currentEdge[0][0] if currentCorner == currentEdge[0][1] else currentEdge[0][1]
                newSpline[0].append(currentCorner)

                activeEdges.remove(currentEdge)
                currentEdge = self.find_edge_from_corner(currentCorner, newSpline[1], activeEdges)

            currentCorner = initialEdge[0][1]
            currentEdge = self.find_edge_from_corner(currentCorner, newSpline[1], activeEdges)

            # If we haven't ended up back where we started, that means the spline isn't closed, and we have to explore the other way
            while currentEdge != None:
                shouldConnect = False
                currentCorner = currentEdge[0][0] if currentCorner == currentEdge[0][1] else currentEdge[0][1]
                newSpline[0].append(currentCorner)

                activeEdges.remove(currentEdge)
                currentEdge = self.find_edge_from_corner(currentCorner, newSpline[1], activeEdges)

            # If a spline has a sufficient number of control points (length)
            if len(newSpline[0]) >= 3:
                if shouldConnect:
                    # Our try to close the loop
                    # To smooth the junction, we duplicate the first and last control points at the opposite end.
                    last = newSpline[0][-1]
                    previousLast = newSpline[0][-2]

                    newSpline[0].append(newSpline[0][0])
                    newSpline[0].append(newSpline[0][1])
                    newSpline[0].insert(0, last)
                    newSpline[0].insert(0, previousLast)  
                self.splines.append(newSpline)


    ##### Print Functions #####

    def print(self, title: str, color="red", saveFig=False):
        """
        Prints the similarity graph stored, superposed on top of the stored image.
        Stored image format is assumed to be YUV (see `RBG_to_YUV`).
        
        :param
            title: str, the title of the figure to plot.
            color: str, a matplotlib [color string](https://matplotlib.org/2.0.2/api/colors_api.html) (default is "red").
            saveFig: boolean, to save the current figure as a png in the output folder. Default is False.
        """
        plt.figure(figsize=(10, 10))
        plt.title(title)
        for node in self.nodes.values():
            for neighbor in node.neighbors:
                plt.plot((node.position[0], neighbor[0]), (node.position[1], neighbor[1]), color=color, linewidth=1)
        plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_YUV2RGB))
        plt.axis("off")
        if saveFig: plt.savefig(f"output/{title}.png")
        plt.show()


    def print_new_pixels_shape(self, title: str, color="black", showLine=False, saveFig=False):
        """
        Prints the new shapes of the pixel of the image, who are now polygons. based on the corners of the nodes.

        :param
            title: str, name of the figure.
            color: str, a matplotlib [color string](https://matplotlib.org/2.0.2/api/colors_api.html) the color of the perimeter of the the new pixels shapes. Default is "black".
            showLine: boolean, to print the perimeter of the the new pixels shapes. Default is False.
            saveFig: boolean, to save the current figure as a png in the output folder. Default is False.
        """
        plt.figure(figsize=(10, 10 * self.img.shape[0]/self.img.shape[1]))
        plt.title(title)
        for node in self.nodes.values():
            poly = Polygon(node.corners)
            x, y = poly.exterior.xy

            line_width = 0
            if showLine:
                line_width = 0.5

            plt.plot(x, y, color=color, linewidth=line_width)
            pixel_color = node.color / 255
            plt.fill(x, y, color=pixel_color)

        plt.ylim(self.img.shape[0], 0)
        plt.axis("off")
        if saveFig: plt.savefig(f"output/{title}.png")
        plt.show()


    def print_splines(self, title="", showControlPoints=False, saveFig=False):
        """
        Print the different extracted splines of the graph. Based on https://github.com/kawache/Python-B-spline-examples.

        :param
            title: str, name of the figure. Default is no name.
            showControlPoints: boolean, to print the controls points of the splines. Default is False.
            saveFig: boolean, to save the current figure as a png in the output folder. Default is False.
        """
        plt.figure(figsize=(10, 10 * self.img.shape[0]/self.img.shape[1]))
        for spline in self.splines:
            plist = spline[0]
            ctr = np.array(plist)
            x = ctr[:, 0]
            y = ctr[:, 1]

            l = len(x)
            t = np.linspace(0, 1, l-2)
            t = np.append([0, 0], t)
            t = np.append(t, [1, 1])

            tck = [t, [x, y], 2]
            u3 = np.linspace(0, 1, l*2, endpoint=True)
            out = interpolate.splev(u3, tck)

            if showControlPoints: plt.plot(x, y, 'k--', label='Control polygon', marker='o', markerfacecolor='red')
            plt.plot(out[0], out[1], 'b', linewidth=2.0, label='B-spline curve')

        plt.ylim(self.img.shape[0], 0)
        plt.axis("off")
        if saveFig: plt.savefig(f"output/{title}.png")
        plt.show()
