import heapq
import math
import networkx as nx

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
    
    return visited_edges, optimal_path

class ALTPreprocessor:
    def __init__(self, graph, num_landmarks=8):  # Increased default landmarks
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
        subgraph = self.graph.subgraph(largest_component)  # Removed .copy() for efficiency
        nodes = list(subgraph.nodes())

        if not nodes:
            return

        self.landmarks = [nodes[0]]
        landmarks_set = {nodes[0]}

        # Initialize minimum distances with the first landmark
        try:
            initial_dists = nx.shortest_path_length(subgraph, nodes[0], weight='length')
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
                dists = nx.shortest_path_length(subgraph, next_landmark, weight='length')
            except nx.NetworkXNoPath:
                continue

            for node in nodes:
                new_dist = dists.get(node, float('inf'))
                if new_dist < min_dist[node]:
                    min_dist[node] = new_dist

    def _precompute_distances(self):
        reversed_graph = nx.reverse(self.graph)  # Compute once and reuse
        for L in self.landmarks:
            if L not in self.graph:
                continue
            try:
                self.forward_distances[L] = nx.single_source_dijkstra_path_length(self.graph, L, weight='length')
                self.backward_distances[L] = nx.single_source_dijkstra_path_length(reversed_graph, L, weight='length')
            except nx.NetworkXNoPath:
                continue


def bidirectional_alt_query(graph, start, end, preprocessor):
    visited_edges = []
    
    landmarks = preprocessor.landmarks
    fw_dists = preprocessor.forward_distances
    bw_dists = preprocessor.backward_distances

    # Precompute all landmark data for faster access
    forward_data = []
    backward_data = []
    for L in landmarks:
        bwd = bw_dists.get(L, {})
        fwd = fw_dists.get(L, {})
        forward_data.append((
            bwd.get(end, float('inf')),
            fwd.get(end, float('inf')),
            bwd,
            fwd
        ))
        backward_data.append((
            bw_dists.get(L, {}).get(start, float('inf')),
            fw_dists.get(L, {}).get(start, float('inf')),
            bw_dists.get(L, {}),
            fw_dists.get(L, {})
        ))

    # Optimized heuristic functions
    def h_forward(u):
        max_h = 0
        for d_end_L, d_L_end, bwd, fwd in forward_data:
            du = bwd.get(u, float('inf'))
            term1 = du - d_end_L
            lu = fwd.get(u, float('inf'))
            term2 = d_L_end - lu
            max_h = max(max_h, term1, term2)
        return max(max_h, 0)

    def h_backward(u):
        max_h = 0
        for d_start_L, d_L_start, bwd, fwd in backward_data:
            du = bwd.get(u, float('inf'))
            term1 = du - d_start_L
            lu = fwd.get(u, float('inf'))
            term2 = d_L_start - lu
            max_h = max(max_h, term1, term2)
        return max(max_h, 0)

    # Initialize data structures
    forward_queue = []
    heapq.heappush(forward_queue, (h_forward(start), start))
    f_dist = {start: 0}
    f_prev = {}
    
    backward_queue = []
    heapq.heappush(backward_queue, (h_backward(end), end))
    b_dist = {end: 0}
    b_prev = {}
    
    best_cost = float('inf')
    meeting_node = None
    processed_f = set()
    processed_b = set()
    
    while forward_queue and backward_queue:
        # Forward search
        f_prio, u = heapq.heappop(forward_queue)
        if f_prio > best_cost:
            break
        if u in processed_f:
            continue
        processed_f.add(u)

        if u in processed_b:
            new_cost = f_dist[u] + b_dist[u]
            if new_cost < best_cost:
                best_cost = new_cost
                meeting_node = u
        
        for v in graph.neighbors(u):
            edge_length = min(data['length'] for data in graph[u][v].values())
            new_g = f_dist[u] + edge_length
            if new_g < f_dist.get(v, float('inf')):
                f_dist[v] = new_g
                f_prev[v] = u
                heapq.heappush(forward_queue, (new_g + h_forward(v), v))
                visited_edges.append((u, v))
        
        # Backward search
        b_prio, u = heapq.heappop(backward_queue)
        if b_prio > best_cost:
            break
        if u in processed_b:
            continue
        processed_b.add(u)
        
        if u in processed_f:
            new_cost = f_dist[u] + b_dist[u]
            if new_cost < best_cost:
                best_cost = new_cost
                meeting_node = u
        
        for v in graph.predecessors(u):
            edge_length = min(data['length'] for data in graph[v][u].values())
            new_g = b_dist[u] + edge_length
            if new_g < b_dist.get(v, float('inf')):
                b_dist[v] = new_g
                b_prev[v] = u
                heapq.heappush(backward_queue, (new_g + h_backward(v), v))
                visited_edges.append((v, u))
        
        # Early termination check
        if forward_queue and backward_queue:
            f_min = forward_queue[0][0]
            b_min = backward_queue[0][0]
            if f_min + b_min >= best_cost:
                break
    
    # Path reconstruction
    path = []
    if meeting_node is not None:
        # Forward path
        node = meeting_node
        while node in f_prev:
            path.append(node)
            node = f_prev[node]
        path.reverse()
        
        # Backward path
        node = meeting_node
        while node in b_prev:
            node = b_prev[node]
            path.append(node)
    
    return visited_edges, path