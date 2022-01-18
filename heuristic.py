from typing import Dict
from definitions import Node, Position


def curves_heuristic(nodes: Dict[Position, Node], position: Position):
    """
    Evaluates the weights of a problematic 2x2 region, using the curves heuristic from the paper.

    :param
        nodes: Dict[Position, Node], a map of the pixels in the similarity graph evaluated. The key represents the position of the node value.
        position: Position, the position of the top left pixel of the problematic region to evaluate. For this problematic region to be valid,
        a connection must be present between the top left and bottom right nodes, as well as betwee the top right and bottom left nodes.
    
    :return
        weight: Tuple[int, int]: the weights of the evaluated edges. The first element of weights applies to the diagonal from top left to bottom right,
        and the second from top right to bottom left.
    """

    # Exploring the similarity graph on the first edge, and from the top left node, going away from the bottom right node, and counting each pixel encountered.
    first_length = 0
    current_node = nodes[position]
    previous_node = nodes[(position[0] + 1, position[1] + 1)]
    while len(current_node.neighbors) == 2 and current_node.position != (position[0] + 1, position[1] + 1):  # Valence-2 node
        tmp = current_node
        current_node = nodes[current_node.neighbors[0]] if nodes[current_node.neighbors[0]].position is not previous_node.position else nodes[current_node.neighbors[1]]
        previous_node = tmp
        first_length += 1

    # Exploring the similarity graph on the first edge, and from the bottom right node, going away from the top left node, and counting each pixel encountered,
    # while making sure to keep using the same counter as before to keep track of the total correctly.
    current_node = nodes[(position[0] + 1, position[1] + 1)]
    previous_node = nodes[position]
    while len(current_node.neighbors) == 2 and current_node.position != position:  # Valence-2 node
        tmp = current_node
        current_node = nodes[current_node.neighbors[0]] if nodes[current_node.neighbors[0]].position is not previous_node.position else nodes[current_node.neighbors[1]]
        previous_node = tmp
        first_length += 1

    # Exploring the similarity graph on the second edge, and from the bottom left node, going away from the top right node, and counting each pixel encountered.
    second_length = 0
    current_node = nodes[(position[0], position[1] + 1)]
    previous_node = nodes[(position[0] + 1, position[1])]
    while len(current_node.neighbors) == 2 and current_node.position != (position[0] + 1, position[1]):  # Valence-2 node
        tmp = current_node
        current_node = nodes[current_node.neighbors[0]] if nodes[current_node.neighbors[0]].position is not previous_node.position else nodes[current_node.neighbors[1]]
        previous_node = tmp
        second_length += 1

    # Exploring the similarity graph on the second edge, and from the top right node, going away from the bottom left node, and counting each pixel encountered,
    # while making sure to keep using the same counter as before to keep track of the total correctly.
    current_node = nodes[(position[0] + 1, position[1])]
    previous_node = nodes[(position[0], position[1] + 1)]
    while len(current_node.neighbors) == 2 and current_node.position != (position[0], position[1] + 1):  # Valence-2 node
        tmp = current_node
        current_node = nodes[current_node.neighbors[0]] if nodes[current_node.neighbors[0]].position is not previous_node.position else nodes[current_node.neighbors[1]]
        previous_node = tmp
        second_length += 1

    return (first_length, second_length)


def sparses_heuristic(nodes: Dict[Position, Node], position: Position):
    """
    Evaluates the weights of a problematic 2x2 region, using the sparse pixels heuristic from the paper.

    :param
        nodes: Dict[Position, Node], a map of the pixels in the similarity graph evaluated. The key represents the position of the node value.
        position: Position, the position of the top left pixel of the problematic region to evaluate. For this problematic region to be valid,
        a connection must be present between the top left and bottom right nodes, as well as betwee the top right and bottom left nodes.
    
    :return
        weight: Tuple[int, int]: the weights of the evaluated edges. The first element of weights applies to the diagonal from top left to bottom right,
        and the second from top right to bottom left.
    """

    # First diagonal
    first_count = 0
    # Work stack
    node_stack = [nodes[position]]
    unique_positions = set()
    while len(node_stack) != 0:
        current_node = node_stack.pop()
        if current_node.position in unique_positions:
            # Skipping previously explored nodes
            continue

        for neighbor_pos in current_node.neighbors:
            # Only considering neighbors in a 8x8 region around the crossing edges. Because "position" is referring to the top left pixel, we have an asymmetry.
            if neighbor_pos[0] < position[0] - 3 or neighbor_pos[0] > position[0] + 4:
                continue
            if neighbor_pos[1] < position[1] - 3 or neighbor_pos[1] > position[1] + 4:
                continue
            if neighbor_pos in unique_positions:
                continue

            node_stack.append(nodes[neighbor_pos])

        first_count += 1
        unique_positions.add(current_node.position)

    # Now doing the same process, but for the second diagonal.
    second_count = 0
    node_stack = [nodes[(position[0], position[1] + 1)]]
    unique_positions = set()
    while len(node_stack) != 0:
        current_node = node_stack.pop()
        if current_node.position in unique_positions:
            continue

        for neighbor_pos in current_node.neighbors:
            if neighbor_pos[0] < position[0] - 3 or neighbor_pos[0] > position[0] + 4:
                continue
            if neighbor_pos[1] < position[1] - 3 or neighbor_pos[1] > position[1] + 4:
                continue
            if neighbor_pos in unique_positions:
                continue

            node_stack.append(nodes[neighbor_pos])

        second_count += 1
        unique_positions.add(current_node.position)

    # The weight is defined as the difference of the values computed.
    weight = abs(first_count - second_count)

    # We only want to give weight to the edge that connects pixels of the least frequent color.
    # The other edge gets 0. In case of equality, both get 0 (as the weight is 0).
    if first_count >= second_count:
        return (0, weight)
    elif first_count < second_count:
        return (weight, 0)


def island_heuristic(nodes: Dict[Position, Node], position: Position):
    """
    Evaluates the weights of a problematic 2x2 region, using the islands heuristic from the paper.

    :param
        nodes: Dict[Position, Node], a map of the pixels in the similarity graph evaluated. The key represents the position of the node value.
        position: Position, the position of the top left pixel of the problematic region to evaluate. For this problematic region to be valid,
        a connection must be present between the top left and bottom right nodes, as well as betwee the top right and bottom left nodes.
    
    :return
        weight: Tuple[int, int]: the weights of the evaluated edges. The first element of weights applies to the diagonal from top left to bottom right,
        and the second from top right to bottom left.
    """

    # We simply add a penalty of 5 (values defined extracted from the paper), to an edge if either of the pixels it connects are only connected by this edge.
    WEIGHT_PENALTY = 5
    
    first_weight = 0
    if len(nodes[position].neighbors) == 1 or len(nodes[(position[0] + 1, position[1] + 1)].neighbors) == 1:
        first_weight = WEIGHT_PENALTY

    second_weight = 0
    if len(nodes[(position[0], position[1] + 1)].neighbors) == 1 or len(nodes[(position[0] + 1, position[1])].neighbors) == 1:
        second_weight = WEIGHT_PENALTY
    return (first_weight, second_weight)


def calculate_heuristics(nodes: Dict[Position, Node], position: Position):
    """
    Returns the weights of the heuritics for both edges.
    Weight 1 is for (x, y)     -> (x + 1, y + 1);
    Weight 2 is for (x, y + 1) -> (x + 1, y);
        
    :param
        nodes: Dict[Position, Node], a map of the pixels in the similarity graph evaluated. The key represents the position of the node value.
        position: Position, the position of the top left pixel of the problematic region to evaluate. For this problematic region to be valid,
        a connection must be present between the top left and bottom right nodes, as well as betwee the top right and bottom left nodes.
        
    :return
        weight: Tuple[int, int]: the total weights of the evaluated edges. The first element of weights applies to the diagonal from top left to bottom right,
        and the second from top right to bottom left.
    """
    curves = curves_heuristic(nodes, position)
    spares = sparses_heuristic(nodes, position)
    islands = island_heuristic(nodes, position)

    return (curves[0] + spares[0] + islands[0], curves[1] + spares[1] + islands[1])
