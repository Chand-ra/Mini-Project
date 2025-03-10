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

def bidirectional_dijkstra(G, start_node, end_node):
    return 0;

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

def bidirectional_astar(G, start_node, end_node):
    return 0;

def contraction_hierarchies(G, start_node, end_node):
    return 0;

def final_algorithm(G, start_node, end_node):
    return 0;