import osmnx as ox
from animation import Animator
from animate_bidAstar import bidAstar_Animator
from algorithms import dijkstra, bidirectional_astar

# ---------------------------
# Load and Prepare the Graph
# ---------------------------
G = ox.graph_from_bbox([72.75, 21.13, 72.87, 21.22]) #subset of Surat
start_node = ox.distance.nearest_nodes(G, 72.7865, 21.1634) #coordinates of SVNIT
end_node = ox.distance.nearest_nodes(G, 72.8410, 21.2055) #coordinates of Surat Railway Station

animator = Animator(G, start_node, end_node)

animator_bidAstar = bidAstar_Animator(G, start_node, end_node) # for bid-A*

animator.animate_path(dijkstra) # to get animation of dijikstra

animator_bidAstar.animate_path(bidirectional_astar) # to get animation of bid-A*

animator_bidAstar.save_animation(bidirectional_astar, "bid_astar_animation.mp4", fps=60)

