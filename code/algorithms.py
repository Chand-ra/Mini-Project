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

def astar(G, start_node, end_node):
    return 0;


#-----------------------------------------------------------------------------------------------------------------

def euclidean_distance(node1, node2, graph):
    x1, y1 = graph.nodes[node1]['x'], graph.nodes[node1]['y']
    x2, y2 = graph.nodes[node2]['x'], graph.nodes[node2]['y']
    return math.hypot(x2 - x1, y2 - y1)

def bidirectional_astar(graph, start, end):
    visited_nodes = []
    visited_edges_fwd = []
    visited_edges_bwd = []
    optimal_path = []

    f_queue = []
    heapq.heappush(f_queue, (0, start, [start]))
    f_visited = {}
    f_g_score = {start: 0}

    b_queue = []
    heapq.heappush(b_queue, (0, end, [end]))
    b_visited = {}
    b_g_score = {end: 0}

    meeting_node = None

    while f_queue and b_queue:
        f_cost, f_current, f_path = heapq.heappop(f_queue)
        visited_nodes.append(f_current)
        f_visited[f_current] = f_path

        if f_current in b_visited:
            meeting_node = f_current
            break

        for neighbor in graph.neighbors(f_current):
            tentative_g = f_g_score[f_current] + graph[f_current][neighbor].get('weight', 1)
            if neighbor not in f_g_score or tentative_g < f_g_score[neighbor]:
                f_g_score[neighbor] = tentative_g
                f_f_cost = tentative_g + euclidean_distance(neighbor, end, graph)
                heapq.heappush(f_queue, (f_f_cost, neighbor, f_path + [neighbor]))
                visited_edges_fwd.append((f_current, neighbor))

        b_cost, b_current, b_path = heapq.heappop(b_queue)
        visited_nodes.append(b_current)
        b_visited[b_current] = b_path

        if b_current in f_visited:
            meeting_node = b_current
            break

        for neighbor in graph.neighbors(b_current):
            tentative_g = b_g_score[b_current] + graph[b_current][neighbor].get('weight', 1)
            if neighbor not in b_g_score or tentative_g < b_g_score[neighbor]:
                b_g_score[neighbor] = tentative_g
                b_f_cost = tentative_g + euclidean_distance(neighbor, start, graph)
                heapq.heappush(b_queue, (b_f_cost, neighbor, b_path + [neighbor]))
                visited_edges_bwd.append((b_current, neighbor))

    if meeting_node:
        f_path = f_visited[meeting_node]
        b_path = b_visited[meeting_node]
        optimal_path = f_path + b_path[::-1][1:]

    return visited_nodes, (visited_edges_fwd, visited_edges_bwd), optimal_path, meeting_node


#-----------------------------------------------------------------------------------------------------------------


def contraction_hierarchies(G, start_node, end_node):
    return 0;

def final_algorithm(G, start_node, end_node):
    return 0;