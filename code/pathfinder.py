import osmnx as ox
from animation import Animator
from algorithms import dijkstra, astar, contraction_hierarchies

# ---------------------------
# Load and Prepare the Graph
# ---------------------------
G = ox.graph_from_bbox([72.75, 21.13, 72.87, 21.22]) #subset of Surat
start_node = ox.distance.nearest_nodes(G, 72.7865, 21.1634) #coordinates of SVNIT
end_node = ox.distance.nearest_nodes(G, 72.8410, 21.2055) #coordinates of Surat Railway Station

animator = Animator(G, start_node, end_node)
animator.animate_path(astar)
#animator.save_animation(astar, "animation.mp4", fps=60)