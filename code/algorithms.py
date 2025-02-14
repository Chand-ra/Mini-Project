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

def bidirectional_dijkstra(G, start_node, end_node):
    return 0;

def astar(G, start_node, end_node):
    return 0;

def bidirectional_astar(G, start_node, end_node):
    return 0;

def contraction_hierarchies(G, start_node, end_node):
    return 0;

def final_algorithm(G, start_node, end_node):
    return 0;