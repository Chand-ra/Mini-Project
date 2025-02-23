import heapq

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

def astar(G, start_node, end_node):
    return 0;

def bidirectional_astar(G, start_node, end_node):
    return 0;

def contraction_hierarchies(G, start_node, end_node):
    return 0;

def final_algorithm(G, start_node, end_node):
    return 0;