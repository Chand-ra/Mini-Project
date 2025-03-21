import osmnx as ox
from animation import Animator
from algorithms import dijkstra, astar, bidirectional_astar, bidirectional_dijkstra, ALTPreprocessor, bidirectional_alt_query

# ---------------------------
# Load and Prepare the Graph
# ---------------------------
G = ox.graph_from_bbox([72.75, 21.13, 72.87, 21.22]) #subset of Surat
start_node = ox.distance.nearest_nodes(G, 72.7865, 21.1634) #coordinates of SVNIT
end_node = ox.distance.nearest_nodes(G, 72.8410, 21.2055) #coordinates of Surat Railway Station

_, path1 = dijkstra(G, start_node, end_node)
visited, path2 = bidirectional_alt_query(G, start_node, end_node, ALTPreprocessor(G))
print(path1 == path2)

animator = Animator(G, start_node, end_node, ALTPreprocessor(G))
animator.animate_path(bidirectional_alt_query)
#animator.save_animation(dijkstra, "animation.mp4", fps=60)