import heapq
import math
import networkx as nx
from collections import defaultdict
    

# The custom algorithm must return visted nodes, visited edges and the optimal path
def dijkstra(graph, start, end):
    visited_edges, optimal_path = [], []

    # Use proper priority queue with (cost, node, path)
    queue = []
    heapq.heappush(queue, (0, start, [start]))
    
    shortest_distance = {node: float('inf') for node in graph.nodes}
    shortest_distance[start] = 0
    while queue:
        current_cost, current_node, current_path = heapq.heappop(queue)  # Proper priority extraction
        if current_cost > shortest_distance[current_node]:
            continue
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
    
    return visited_edges, optimal_path

def bidirectional_dijkstra(graph, start, end):
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
    
    return visited_edges, optimal_path

def astar(graph, start, end):
    visited_edges = []

    nodes = graph.nodes()
    coords = {node: (nodes[node]['x'], nodes[node]['y']) for node in nodes}
    end_x, end_y = coords[end]

    def h(u):
        x, y = coords[u]
        return math.hypot(end_x - x, end_y - y)
    
    # Initialize the open set as a priority queue with (f, node)
    open_queue = []
    heapq.heappush(open_queue, (h(start), start))
    
    # Dictionary to store the cost from start to each node
    shortest_distance = {node: float('inf') for node in nodes}
    shortest_distance[start] = 0
    
    # Dictionary to reconstruct the optimal path
    prev = {}    
    processed = set()
    
    while open_queue:
        current_f, current_node = heapq.heappop(open_queue)
        if current_node in processed:
            continue
        processed.add(current_node)

        # If we've reached the target, break early.
        if current_node == end:
            break
        current_g = shortest_distance[current_node]
        # Process neighbors
        for neighbor in graph.neighbors(current_node):
            # Get the minimum edge length between current_node and neighbor
            edge_length = min(data['length'] for data in graph[current_node][neighbor].values())
            new_g = current_g + edge_length
            if new_g < shortest_distance.get(neighbor, float('inf')):
                shortest_distance[neighbor] = new_g
                prev[neighbor] = current_node
                f_score = new_g + h(neighbor)
                heapq.heappush(open_queue, (f_score, neighbor))
                visited_edges.append((current_node, neighbor))
    
    # Reconstruct the path from start to end, if found.
    optimal_path = []
    if end in processed:
        node = end
        path = []
        while node != start:
            path.append(node)
            node = prev.get(node)
            if node is None:
                break
        path.append(start)
        optimal_path = list(reversed(path))
    
    return visited_edges, optimal_path

def bidirectional_astar(graph, start, end):

    visited_edges = []
    
    # Early exit if start and end are the same
    if start == end:
        return visited_edges, [start]
    
    # Precompute coordinates for all nodes - use generator expression for efficiency
    nodes = graph.nodes()
    coords = {node: (nodes[node]['x'], nodes[node]['y']) for node in nodes}
    
    # Precompute heuristic values for both directions
    end_x, end_y = coords[end]
    start_x, start_y = coords[start]
    
    # Cache heuristic calculations
    forward_heuristic = {}
    backward_heuristic = {}
    
    def h_forward(u):
        if u not in forward_heuristic:
            x, y = coords[u]
            forward_heuristic[u] = math.hypot(end_x - x, end_y - y)
        return forward_heuristic[u]
    
    def h_backward(u):
        if u not in backward_heuristic:
            x, y = coords[u]
            backward_heuristic[u] = math.hypot(x - start_x, y - start_y)
        return backward_heuristic[u]
    
    # Forward search initialization with entry counter for stable sorting
    entry_count = 0
    forward_queue = [(h_forward(start), 0, entry_count, start)]  # (f, g, tie_breaker, node)
    shortest_distance_forward = defaultdict(lambda: float('inf'))
    shortest_distance_forward[start] = 0
    forward_prev = {}
    
    # Backward search initialization
    entry_count += 1
    backward_queue = [(h_backward(end), 0, entry_count, end)]  # (f, g, tie_breaker, node)
    shortest_distance_backward = defaultdict(lambda: float('inf'))
    shortest_distance_backward[end] = 0
    backward_prev = {}
    
    best_total_cost = float('inf')
    meeting_node = None
    processed_forward = set()
    processed_backward = set()
    
    # Precompute edge costs where possible
    edge_costs = {}
    
    def get_edge_cost(u, v):
        if (u, v) not in edge_costs:
            edge_costs[(u, v)] = min(data.get('length', 1) for data in graph[u][v].values())
        return edge_costs[(u, v)]
    
    # Main loop
    while forward_queue and backward_queue:
        # Early termination check
        forward_min_f, forward_min_g = forward_queue[0][0], forward_queue[0][1]
        backward_min_f, backward_min_g = backward_queue[0][0], backward_queue[0][1]
        if forward_min_g + backward_min_g + max(0, forward_min_f - forward_min_g + backward_min_f - backward_min_g) >= best_total_cost:
            break
        
        # Alternate between forward and backward search based on estimated progress
        if forward_min_f <= backward_min_f:
            # Process forward search
            f_current, current_g, _, current_node = heapq.heappop(forward_queue)
            
            if current_node in processed_forward or current_g > shortest_distance_forward[current_node]:
                continue
                
            processed_forward.add(current_node)
            
            # Check backward search results
            if current_node in processed_backward:
                total_cost = current_g + shortest_distance_backward[current_node]
                if total_cost < best_total_cost:
                    best_total_cost = total_cost
                    meeting_node = current_node
            
            # Expand forward neighbors
            for neighbor in graph.neighbors(current_node):
                if neighbor in processed_forward:
                    continue
                    
                edge_length = get_edge_cost(current_node, neighbor)
                new_g = current_g + edge_length
                
                if new_g < shortest_distance_forward[neighbor]:
                    shortest_distance_forward[neighbor] = new_g
                    forward_prev[neighbor] = current_node
                    f_score = new_g + h_forward(neighbor)
                    entry_count += 1
                    heapq.heappush(forward_queue, (f_score, new_g, entry_count, neighbor))
                    visited_edges.append((current_node, neighbor))
                    
                    # Check if this node has been reached in backward search
                    if neighbor in shortest_distance_backward and neighbor != float('inf'):
                        total_cost = new_g + shortest_distance_backward[neighbor]
                        if total_cost < best_total_cost:
                            best_total_cost = total_cost
                            meeting_node = neighbor
        else:
            # Process backward search
            b_current, current_g, _, current_node = heapq.heappop(backward_queue)
            
            if current_node in processed_backward or current_g > shortest_distance_backward[current_node]:
                continue
                
            processed_backward.add(current_node)
            
            # Check forward search results
            if current_node in processed_forward:
                total_cost = shortest_distance_forward[current_node] + current_g
                if total_cost < best_total_cost:
                    best_total_cost = total_cost
                    meeting_node = current_node
            
            # Expand backward predecessors
            for predecessor in graph.predecessors(current_node):
                if predecessor in processed_backward:
                    continue
                    
                edge_length = get_edge_cost(predecessor, current_node)
                new_g = current_g + edge_length
                
                if new_g < shortest_distance_backward[predecessor]:
                    shortest_distance_backward[predecessor] = new_g
                    backward_prev[predecessor] = current_node
                    f_score = new_g + h_backward(predecessor)
                    entry_count += 1
                    heapq.heappush(backward_queue, (f_score, new_g, entry_count, predecessor))
                    visited_edges.append((predecessor, current_node))
                    
                    # Check if this node has been reached in forward search
                    if predecessor in shortest_distance_forward and shortest_distance_forward[predecessor] != float('inf'):
                        total_cost = shortest_distance_forward[predecessor] + new_g
                        if total_cost < best_total_cost:
                            best_total_cost = total_cost
                            meeting_node = predecessor
    
    # Path reconstruction - optimized to avoid unnecessary operations
    if meeting_node is None:
        return visited_edges, []
        
    # Reconstruct path (forward path + backward path)
    forward_path = []
    node = meeting_node
    while node in forward_prev:
        forward_path.append(node)
        node = forward_prev[node]
    forward_path.append(start)
    forward_path.reverse()
    
    # Add backward path
    if meeting_node != end:
        backward_path = []
        node = backward_prev.get(meeting_node)
        while node is not None:
            backward_path.append(node)
            node = backward_prev.get(node)
        return visited_edges, forward_path + backward_path
    
    return visited_edges, forward_path

import heapq
import networkx as nx

class ALTPreprocessor:
    def __init__(self, graph, num_landmarks=8):
        self.graph = graph
        self.num_landmarks = num_landmarks
        self.landmarks = []
        self.forward_distances = {}
        self.backward_distances = {}

        self._select_landmarks()
        self._precompute_distances()

    def _select_landmarks(self):
        if not self.graph:
            return

        components = list(nx.strongly_connected_components(self.graph))
        if not components:
            return
        largest_component = max(components, key=len)
        subgraph = self.graph.subgraph(largest_component)
        nodes = list(subgraph.nodes())

        if not nodes:
            return

        self.landmarks = [nodes[0]]
        landmarks_set = {nodes[0]}

        # Initialize minimum distances with the first landmark
        try:
            initial_dists = nx.shortest_path_length(subgraph, source=nodes[0], weight='length')
        except nx.NetworkXNoPath:
            initial_dists = {}
        min_dist = {node: initial_dists.get(node, float('inf')) for node in nodes}

        while len(self.landmarks) < self.num_landmarks and len(self.landmarks) < len(nodes):
            max_dist = -1
            next_landmark = None

            # Find the node with the maximum minimal distance
            for node in nodes:
                if node in landmarks_set:
                    continue
                current_min = min_dist[node]
                if current_min > max_dist:
                    max_dist = current_min
                    next_landmark = node

            if next_landmark is None:
                break  # No suitable node found

            self.landmarks.append(next_landmark)
            landmarks_set.add(next_landmark)

            # Update min_dist with distances from the new landmark
            try:
                dists = nx.shortest_path_length(subgraph, source=next_landmark, weight='length')
            except nx.NetworkXNoPath:
                continue

            for node in nodes:
                new_dist = dists.get(node, float('inf'))
                if new_dist < min_dist[node]:
                    min_dist[node] = new_dist

    def _precompute_distances(self):
        reversed_graph = nx.reverse(self.graph)
        for L in self.landmarks:
            if L not in self.graph:
                continue
            try:
                self.forward_distances[L] = nx.single_source_dijkstra_path_length(self.graph, L, weight='length')
                self.backward_distances[L] = nx.single_source_dijkstra_path_length(reversed_graph, L, weight='length')
            except nx.NetworkXNoPath:
                continue


def bidirectional_alt_query(graph, start, end, preprocessor):
    """
    Bidirectional ALT algorithm using A* with landmarks and triangle inequality
    """
    visited_edges = []
    
    # Early exit check
    if start == end:
        return visited_edges, [start]
    
    landmarks = preprocessor.landmarks
    fw_dists = preprocessor.forward_distances
    bw_dists = preprocessor.backward_distances

    # Cache all landmark data for faster access
    landmark_cache = []
    for L in landmarks:
        if L not in fw_dists or L not in bw_dists:
            continue
            
        landmark_data = {
            'l_to_end': fw_dists[L].get(end, float('inf')),
            'end_to_l': bw_dists[L].get(end, float('inf')),
            'l_to_start': fw_dists[L].get(start, float('inf')),
            'start_to_l': bw_dists[L].get(start, float('inf')),
            'l_fw': fw_dists[L],
            'l_bw': bw_dists[L]
        }
        landmark_cache.append(landmark_data)

    # Optimized heuristic functions with caching
    forward_h_cache = {}
    backward_h_cache = {}
    
    def h_forward(u):
        if u in forward_h_cache:
            return forward_h_cache[u]
            
        max_h = 0
        for data in landmark_cache:
            # Triangle inequality: d(u,t) >= |d(u,L) - d(t,L)| and d(u,t) >= |d(L,u) - d(L,t)|
            # where L is a landmark, u is the current node, t is the target (end)
            u_to_l = data['l_bw'].get(u, float('inf'))
            l_to_u = data['l_fw'].get(u, float('inf'))
            
            # |d(u,L) - d(t,L)|
            term1 = abs(u_to_l - data['end_to_l'])
            # |d(L,u) - d(L,t)|
            term2 = abs(l_to_u - data['l_to_end'])
            
            max_h = max(max_h, term1, term2)
        
        forward_h_cache[u] = max_h
        return max_h

    def h_backward(u):
        if u in backward_h_cache:
            return backward_h_cache[u]
            
        max_h = 0
        for data in landmark_cache:
            # Triangle inequality: d(s,u) >= |d(s,L) - d(u,L)| and d(s,u) >= |d(L,s) - d(L,u)|
            # where L is a landmark, u is the current node, s is the source (start)
            u_to_l = data['l_bw'].get(u, float('inf'))
            l_to_u = data['l_fw'].get(u, float('inf'))
            
            # |d(s,L) - d(u,L)|
            term1 = abs(data['start_to_l'] - u_to_l)
            # |d(L,s) - d(L,u)|
            term2 = abs(data['l_to_start'] - l_to_u)
            
            max_h = max(max_h, term1, term2)
        
        backward_h_cache[u] = max_h
        return max_h

    # Initialize data structures with entry counter for stable sorting
    entry_count = 0
    forward_queue = []
    h_start = h_forward(start)
    heapq.heappush(forward_queue, (h_start, 0, entry_count, start))  # (f, g, tiebreaker, node)
    forward_dist = {start: 0}
    forward_prev = {}
    
    entry_count += 1
    backward_queue = []
    h_end = h_backward(end)
    heapq.heappush(backward_queue, (h_end, 0, entry_count, end))  # (f, g, tiebreaker, node)
    backward_dist = {end: 0}
    backward_prev = {}
    
    best_cost = float('inf')
    meeting_node = None
    processed_forward = set()
    processed_backward = set()
    
    # Edge cost cache
    edge_costs = {}
    
    def get_edge_cost(u, v):
        if (u, v) not in edge_costs:
            edge_costs[(u, v)] = min(data.get('length', 1) for data in graph[u][v].values())
        return edge_costs[(u, v)]
    
    while forward_queue and backward_queue:
        # Early termination check
        if forward_queue and backward_queue:
            f_min_f, f_min_g = forward_queue[0][0], forward_queue[0][1]
            b_min_f, b_min_g = backward_queue[0][0], backward_queue[0][1]
            if f_min_g + b_min_g >= best_cost:
                break
        
        # Process forward search
        if forward_queue and (not backward_queue or forward_queue[0][0] <= backward_queue[0][0]):
            f_score, g_score, _, current = heapq.heappop(forward_queue)
            
            if current in processed_forward or g_score > forward_dist[current]:
                continue
                
            processed_forward.add(current)
            
            # Check if we found a meeting point
            if current in backward_dist:
                total_cost = g_score + backward_dist[current]
                if total_cost < best_cost:
                    best_cost = total_cost
                    meeting_node = current
            
            # Expand neighbors
            for neighbor in graph.neighbors(current):
                if neighbor in processed_forward:
                    continue
                    
                edge_length = get_edge_cost(current, neighbor)
                new_g = g_score + edge_length
                
                if new_g < forward_dist.get(neighbor, float('inf')):
                    forward_dist[neighbor] = new_g
                    forward_prev[neighbor] = current
                    h_value = h_forward(neighbor)
                    entry_count += 1
                    heapq.heappush(forward_queue, (new_g + h_value, new_g, entry_count, neighbor))
                    visited_edges.append((current, neighbor))
                    
                    # Check if this is a meeting point
                    if neighbor in backward_dist:
                        total_cost = new_g + backward_dist[neighbor]
                        if total_cost < best_cost:
                            best_cost = total_cost
                            meeting_node = neighbor
        
        # Process backward search
        elif backward_queue:
            b_score, g_score, _, current = heapq.heappop(backward_queue)
            
            if current in processed_backward or g_score > backward_dist[current]:
                continue
                
            processed_backward.add(current)
            
            # Check if we found a meeting point
            if current in forward_dist:
                total_cost = forward_dist[current] + g_score
                if total_cost < best_cost:
                    best_cost = total_cost
                    meeting_node = current
            
            # Expand predecessors
            for predecessor in graph.predecessors(current):
                if predecessor in processed_backward:
                    continue
                    
                edge_length = get_edge_cost(predecessor, current)
                new_g = g_score + edge_length
                
                if new_g < backward_dist.get(predecessor, float('inf')):
                    backward_dist[predecessor] = new_g
                    backward_prev[predecessor] = current
                    h_value = h_backward(predecessor)
                    entry_count += 1
                    heapq.heappush(backward_queue, (new_g + h_value, new_g, entry_count, predecessor))
                    visited_edges.append((predecessor, current))
                    
                    # Check if this is a meeting point
                    if predecessor in forward_dist:
                        total_cost = forward_dist[predecessor] + new_g
                        if total_cost < best_cost:
                            best_cost = total_cost
                            meeting_node = predecessor
    
    # Path reconstruction
    if meeting_node is None:
        return visited_edges, []
        
    # Reconstruct forward path
    forward_path = []
    node = meeting_node
    while node in forward_prev:
        forward_path.append(node)
        node = forward_prev[node]
    forward_path.append(start)  # Add start node
    forward_path.reverse()
    
    # Reconstruct backward path if meeting node is not the end
    if meeting_node != end:
        node = backward_prev.get(meeting_node)
        while node is not None:
            forward_path.append(node)
            node = backward_prev.get(node)
    
    return visited_edges, forward_path