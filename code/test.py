import algorithms
import networkx as nx
import osmnx as ox
import random

def calculate_path_length(G, path):
    total = 0
    for i in range(len(path) - 1):
        # For MultiGraphs, take the first edge (adjust if needed)
        data = G[path[i]][path[i+1]][0]
        total += data.get('length', 1)
    return total

# Test function to compare your custom algorithm against NetworkX's shortest path
def compare_paths(graph, start, end, algorithm):
    _, custom_optimal_path = algorithm(graph, start, end)
    nx_optimal_path = nx.shortest_path(graph, source=start, target=end, weight='length')
    
    # Calculate the total cost for both paths
    custom_cost = calculate_path_length(graph, custom_optimal_path)
    nx_cost = calculate_path_length(graph, nx_optimal_path)
    
    if custom_optimal_path != nx_optimal_path or abs(custom_cost - nx_cost) > 1e-6:
        print("Test failed, start =", start, ",end =", end, ": The algorithm's path does not match NetworkX's.")
        print("Path cost in algorithm: ", custom_cost, ", Path cost in NetwokX: ", nx_cost)

def test_algorithm(algorithm):
    G = ox.graph_from_bbox([72.82, 21.16, 72.84, 21.18])
    nodes_list = list(G.nodes)
    for i in range(0, 1000):
        try:
            compare_paths(G, random.choice(nodes_list), random.choice(nodes_list), algorithm)
        except:
            nx.exception.NetworkXNoPath

# ---------------------------
# Driver Code
# ---------------------------
test_algorithm(algorithms.dijkstra)
#test_algorithm(algorithms.astar)