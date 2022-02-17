import csv
import numpy as np
import argparse
from functions import *
from pathlib import Path

dir_name = "analysis_data"

Path(dir_name).mkdir(parents=True, exist_ok=True)

# The text file has 378142421 lines

# Instantiate the parser
parser = argparse.ArgumentParser(description="Parser for the number of the sample")
# Number of the sample
parser.add_argument(
    "--nsample", type=int, help="Number of the sample we will parse", default=0
)

args = parser.parse_args()
print(args.nsample)
G = nx.DiGraph()
lines_arr = np.genfromtxt("samples/sample_{}.txt".format(args.nsample))
for n, p in lines_arr:
    G.add_edge(int(n), int(p))
# nx.draw(G, font_weight="bold")
# plt.show()

print("These are the parameters parameters for the {}-th sample".format(args.nsample))
print("The density of the network is: ", density_net(G))
print("The average degree of the network is: ", average_degree(G))
print("The probability for a node to have degree 0.3 is: ", degree_distribution(G, 0.3))
print(
    "The probability for a node to have clustering degree 0.1 is: ",
    clustering_degree(G, 0.1),
)
degree_distribution_plot(G, type0=0, name0=args.nsample, path0=dir_name)
degree_distribution_plot(G, type0=1, name0=args.nsample, path0=dir_name)
degree_distribution_plot(G, type0=2, name0=args.nsample, path0=dir_name)
cumulative_degree_distribution_plot(G, type0=0, name0=args.nsample, path0=dir_name)
cumulative_degree_distribution_plot(G, type0=1, name0=args.nsample, path0=dir_name)
cumulative_degree_distribution_plot(G, type0=2, name0=args.nsample, path0=dir_name)
lorentz_curve_plot(G, type0=0, name0=args.nsample, path0=dir_name)
lorentz_curve_plot(G, type0=1, name0=args.nsample, path0=dir_name)
lorentz_curve_plot(G, type0=2, name0=args.nsample, path0=dir_name)
clustering_degree_plot(G, name0=args.nsample, path0=dir_name)
betweenness_centrality_plot(G, name0=args.nsample, path0=dir_name)
degree_correlation_plot(
    G, source0="out", target0="out", name0=args.nsample, path0=dir_name
)
degree_correlation_plot(
    G, source0="in", target0="out", name0=args.nsample, path0=dir_name
)
degree_correlation_plot(
    G, source0="out", target0="in", name0=args.nsample, path0=dir_name
)
degree_correlation_plot(
    G, source0="in", target0="in", name0=args.nsample, path0=dir_name
)
k_cores(G, name0=args.nsample, path0=dir_name)
diameter(G)
shortest_path(G, name0=args.nsample, path0=dir_name)
random_walk(G, name0=args.nsample, path0=dir_name)
