from typing import Tuple

Position = Tuple[int, int]          # (x, y)
Edge = Tuple[Position, Position]    # (x1, y1), (x2, y2)
Color = Tuple[int, int, int]        # (R, G, B) or (Y, U, V)

class Node:
    def __init__(self, x: int, y: int, color: Tuple[int, int, int]):
        self.position = (x, y)
        self.color = color
        self.neighbors = []
        self.corners = []

    def __str__(self):
        position = "\n Position : (%s, %s)" % (self.position[0], self.position[1])
        color = "\n YUV : (%s, %s, %s)" % (self.color[0], self.color[1], self.color[2])

        neighbors = "\n Neighbors (%s): " % (len(self.neighbors))
        for n in self.neighbors:
            neighbors += "\n (%s, %s)" % (n[0], n[1])

        return "Node" + position + color + neighbors
