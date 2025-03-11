import heapq
import math

# The custom algorithm must return visted nodes, visited edges and the optimal path
def dijkstra(graph, start, end):
    visited_nodes, visited_edges, optimal_path = [], [], []

    # Use proper priority queue with (cost, node, path)
    queue = []
    heapq.heappush(queue, (0, start, [start]))
    
    shortest_distance = {node: float('inf') for node in graph.nodes}
    shortest_distance[start] = 0
    while queue:
        current_cost, current_node, current_path = heapq.heappop(queue)  # Proper priority extraction
        if current_cost > shortest_distance[current_node]:
            continue
        visited_nodes.append(current_node)
        if current_node == end:
            optimal_path = current_path
            break
        # Process neighbors in sorted order for deterministic behavior
        for neighbor in sorted(graph.neighbors(current_node), key=lambda x: graph[current_node][x][0]['length']):
            edge_costs = [data.get('length', 1) for data in graph[current_node][neighbor].values()]
            min_edge_cost = min(edge_costs)
            total_cost = current_cost + min_edge_cost

            if total_cost < shortest_distance[neighbor]:
                shortest_distance[neighbor] = total_cost
                heapq.heappush(queue, (total_cost, neighbor, current_path + [neighbor]))
                # Record edge only when it's actually used in the optimal path
                visited_edges.append((current_node, neighbor))
    
    return visited_nodes, visited_edges, optimal_path

def bidirectional_dijkstra(graph, start, end):
    visited_nodes = []
    visited_edges = []
    
    # Forward search initialization
    forward_queue = [(0, start)]
    shortest_distance_forward = {node: float('inf') for node in graph.nodes}
    shortest_distance_forward[start] = 0
    forward_prev = {node: None for node in graph.nodes}
    
    # Backward search initialization
    backward_queue = [(0, end)]
    shortest_distance_backward = {node: float('inf') for node in graph.nodes}
    shortest_distance_backward[end] = 0
    backward_prev = {node: None for node in graph.nodes}
    
    best_total_cost = float('inf')
    meeting_node = None
    
    while forward_queue or backward_queue:
        if not forward_queue:
            process_forward = False
        elif not backward_queue:
            process_forward = True
        else:
            process_forward = forward_queue[0][0] <= backward_queue[0][0]
        
        if process_forward:
            current_cost, current_node = heapq.heappop(forward_queue)
            if current_cost > shortest_distance_forward[current_node]:
                continue
            visited_nodes.append(current_node)
            
            # Check if backward search has reached current_node
            if shortest_distance_backward[current_node] != float('inf'):
                total_cost = current_cost + shortest_distance_backward[current_node]
                if total_cost < best_total_cost:
                    best_total_cost = total_cost
                    meeting_node = current_node
            
            # Process forward neighbors in sorted order to preserve edge visitation order
            for neighbor in sorted(graph.neighbors(current_node),
                                   key=lambda x: graph[current_node][x][0]['length']):
                edge_data = graph[current_node][neighbor]
                min_edge_cost = min(data.get('length', 1) for data in edge_data.values())
                new_cost = current_cost + min_edge_cost
                if new_cost < shortest_distance_forward[neighbor]:
                    shortest_distance_forward[neighbor] = new_cost
                    forward_prev[neighbor] = current_node
                    heapq.heappush(forward_queue, (new_cost, neighbor))
                    visited_edges.append((current_node, neighbor))
        else:
            current_cost, current_node = heapq.heappop(backward_queue)
            if current_cost > shortest_distance_backward[current_node]:
                continue
            visited_nodes.append(current_node)
            
            # Check if forward search has reached current_node
            if shortest_distance_forward[current_node] != float('inf'):
                total_cost = shortest_distance_forward[current_node] + current_cost
                if total_cost < best_total_cost:
                    best_total_cost = total_cost
                    meeting_node = current_node
            
            # Process backward predecessors in sorted order to preserve edge visitation order
            for predecessor in sorted(graph.predecessors(current_node),
                                      key=lambda x: graph[x][current_node][0]['length']):
                edge_data = graph[predecessor][current_node]
                min_edge_cost = min(data.get('length', 1) for data in edge_data.values())
                new_cost = current_cost + min_edge_cost
                if new_cost < shortest_distance_backward[predecessor]:
                    shortest_distance_backward[predecessor] = new_cost
                    backward_prev[predecessor] = current_node
                    heapq.heappush(backward_queue, (new_cost, predecessor))
                    visited_edges.append((current_node, predecessor))
        
        current_forward_min = forward_queue[0][0] if forward_queue else float('inf')
        current_backward_min = backward_queue[0][0] if backward_queue else float('inf')
        if current_forward_min + current_backward_min >= best_total_cost:
            break
    
    # Reconstruct the optimal path if a meeting point was found
    optimal_path = []
    if meeting_node is not None:
        # Build forward path from start to meeting_node
        node = meeting_node
        forward_path = []
        while node is not None:
            forward_path.append(node)
            node = forward_prev[node]
        forward_path.reverse()
        
        # Build backward path from meeting_node to end
        node = backward_prev[meeting_node]
        backward_path = []
        while node is not None:
            backward_path.append(node)
            node = backward_prev[node]
        
        optimal_path = forward_path + backward_path
    
    return visited_nodes, visited_edges, optimal_path

def astar(graph, start, end):
    visited_nodes, visited_edges, optimal_path = [], [], []

    def manhattan_distance(node1, node2):
    # Assume nodes have 'x' and 'y' coordinates in the graph
        x1, y1 = graph.nodes[node1]['x'], graph.nodes[node1]['y']
        x2, y2 = graph.nodes[node2]['x'], graph.nodes[node2]['y']
        return abs(x1 - x2) + abs(y1 - y2)

    # Priority queue stores (f_score, g_score, node, path)
    queue = []
    initial_g = 0
    initial_h = manhattan_distance(start, end)
    heapq.heappush(queue, (initial_g + initial_h, initial_g, start, [start]))
    
    shortest_distance = {node: float('inf') for node in graph.nodes}
    shortest_distance[start] = 0
    
    while queue:
        current_f, current_g, current_node, current_path = heapq.heappop(queue)
        
        # Skip if we've already found a better path to this node
        if current_g > shortest_distance[current_node]:
            continue
            
        visited_nodes.append(current_node)
        if current_node == end:
            optimal_path = current_path
            break
        
        # Process neighbors in sorted order for deterministic behavior
        for neighbor in sorted(graph.neighbors(current_node), key=lambda x: graph[current_node][x][0]['length']):
            edge_costs = [data.get('length', 1) for data in graph[current_node][neighbor].values()]
            min_edge_cost = min(edge_costs)
            tentative_g = current_g + min_edge_cost
            
            if tentative_g < shortest_distance[neighbor]:
                shortest_distance[neighbor] = tentative_g
                new_path = current_path + [neighbor]
                f_score = tentative_g + manhattan_distance(neighbor, end)
                heapq.heappush(queue, (f_score, tentative_g, neighbor, new_path))
                visited_edges.append((current_node, neighbor))
    
    return visited_nodes, visited_edges, optimal_path

def bidirectional_astar(graph, start, end):
    visited_nodes = []
    visited_edges = []
    
    # Precompute coordinates for all nodes
    nodes = graph.nodes()
    coords = {node: (nodes[node]['x'], nodes[node]['y']) for node in nodes}
    
    # Precompute heuristic values for both directions
    end_x, end_y = coords[end]
    start_x, start_y = coords[start]
    
    def h_forward(u):
        x, y = coords[u]
        return math.hypot(end_x - x, end_y - y)
    
    def h_backward(u):
        x, y = coords[u]
        return math.hypot(start_x - x, start_y - y)
    
    # Forward search initialization
    forward_queue = []
    heapq.heappush(forward_queue, (h_forward(start), start))
    shortest_distance_forward = {node: float('inf') for node in nodes}
    shortest_distance_forward[start] = 0
    forward_prev = {}
    
    # Backward search initialization
    backward_queue = []
    heapq.heappush(backward_queue, (h_backward(end), end))
    shortest_distance_backward = {node: float('inf') for node in nodes}
    shortest_distance_backward[end] = 0
    backward_prev = {}
    
    best_total_cost = float('inf')
    meeting_node = None
    processed_forward = set()
    processed_backward = set()
    
    while forward_queue and backward_queue:
        # Process forward search
        f_current_f, current_node = heapq.heappop(forward_queue)
        if f_current_f > best_total_cost:
            break
        
        if current_node in processed_forward:
            continue
        processed_forward.add(current_node)
        visited_nodes.append(current_node)
        
        current_g = shortest_distance_forward[current_node]
        
        # Check backward search results
        if current_node in processed_backward:
            total_cost = current_g + shortest_distance_backward[current_node]
            if total_cost < best_total_cost:
                best_total_cost = total_cost
                meeting_node = current_node
        
        # Expand forward neighbors
        for neighbor in graph.neighbors(current_node):
            edge_length = min(data['length'] for data in graph[current_node][neighbor].values())
            new_g = current_g + edge_length
            if new_g < shortest_distance_forward.get(neighbor, float('inf')):
                shortest_distance_forward[neighbor] = new_g
                forward_prev[neighbor] = current_node
                f_score = new_g + h_forward(neighbor)
                heapq.heappush(forward_queue, (f_score, neighbor))
                visited_edges.append((current_node, neighbor))
        
        # Process backward search
        b_current_f, current_node = heapq.heappop(backward_queue)
        if b_current_f > best_total_cost:
            break
        
        if current_node in processed_backward:
            continue
        processed_backward.add(current_node)
        visited_nodes.append(current_node)
        
        current_g = shortest_distance_backward[current_node]
        
        # Check forward search results
        if current_node in processed_forward:
            total_cost = shortest_distance_forward[current_node] + current_g
            if total_cost < best_total_cost:
                best_total_cost = total_cost
                meeting_node = current_node
        
        # Expand backward predecessors
        for predecessor in graph.predecessors(current_node):
            edge_length = min(data['length'] for data in graph[predecessor][current_node].values())
            new_g = current_g + edge_length
            if new_g < shortest_distance_backward.get(predecessor, float('inf')):
                shortest_distance_backward[predecessor] = new_g
                backward_prev[predecessor] = current_node
                f_score = new_g + h_backward(predecessor)
                heapq.heappush(backward_queue, (f_score, predecessor))
                visited_edges.append((predecessor, current_node))
        
        # Early termination check
        if forward_queue and backward_queue:
            forward_min = forward_queue[0][0]
            backward_min = backward_queue[0][0]
            if forward_min + backward_min >= best_total_cost:
                break
    
    # Path reconstruction (same as before)
    optimal_path = []
    if meeting_node is not None:
        # Forward path
        path = []
        node = meeting_node
        while node is not None:
            path.append(node)
            node = forward_prev.get(node)
        path.reverse()
        
        # Backward path
        node = backward_prev.get(meeting_node)
        while node is not None:
            path.append(node)
            node = backward_prev.get(node)
        
        optimal_path = path
    
    return visited_nodes, visited_edges, optimal_path


def preprocess_contraction_hierarchy(G):
    return 0;

def final_algorithm(G, start_node, end_node):
    return 0;