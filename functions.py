import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import csv


def density_net(G0):
    """
    Function that receives a directed network and computes the density of it
    """
    m0 = G0.size()  # Number of links in the network G0
    n0 = G0.number_of_nodes()  # Number of nodes in the network G0
    return m0 / (n0 * (n0 - 1.0))


def average_degree(G0):
    """
    Function that receives a directed network and computes the average degree of it
    """
    ni = np.asarray(G0.degree())[
        :, 0
    ]  # numpy array containing the identification of each node on the network G0
    ki = np.asarray(G0.degree())[
        :, 1
    ]  # numpy array containing the degree of each node on the network G0
    return np.average(ki)


def degree_distribution(G0, k0):
    """
    Function that receives a directed network and a degree k0 and computes the probability that a node has degree k0
    """
    ni = np.asarray(G0.degree())[
        :, 0
    ]  # numpy array containing the identification of each node on the network G0
    ki = np.asarray(G0.degree())[
        :, 1
    ]  # numpy array containing the degree of each node on the network G0
    n0 = G0.number_of_nodes()  # Number of nodes in the network G0
    n_k0 = ki[ki == k0]  # numpy array containing the amount of nodes with degree k0
    ni_k0 = ni[ki == k0]  # numpy array containing the id of nodes with degree k0
    return np.size(ni_k0) / n0


def degree_distribution_plot(
    G0, type0=0, bins0=20, name0="Default sample name", path0="default_path"
):
    """
    Function that receives a directed network, the type of degree (0 for all nodes, 1 for in
    nodes and 2 for out nodes and the number of bins (20 by default) and computes distribution
    of the degrees and plots it
    """
    try:
        assert (
            type0 == 0 or type0 == 1 or type0 == 2
        )  # Tests that the clustering degree is in fact a number between 0 and 1
    except AssertionError:
        print(
            "Input type should be either 0 for all nodes, 1 for in nodes or 2 for out nodes"
        )
    else:
        if type0 == 0:
            x0 = np.asarray(G0.degree())[:, 0]
            y0 = np.asarray(G0.degree())[:, 1]
            plt.xlabel("Degree (d)")
            nametype = "all nodes"
            saving_name = "all_nodes"
        elif type0 == 1:
            x0 = np.asarray(G0.in_degree())[:, 0]
            y0 = np.asarray(G0.in_degree())[:, 1]
            plt.xlabel(r"Indegree (d$^-$)")
            nametype = "in degree nodes"
            saving_name = "in_nodes"
        else:
            x0 = np.asarray(G0.out_degree())[:, 0]
            y0 = np.asarray(G0.out_degree())[:, 1]
            plt.xlabel(r"Outdegree (d$^+$)")
            nametype = "out degree nodes"
            saving_name = "out_nodes"
        counts0, bines0, nn0 = plt.hist(y0, bins=bins0)
        plt.yscale("log")
        plt.ylabel("Frequency")
        plt.title(
            "Degree distribution plot for sample {} for {}".format(name0, nametype)
        )
        plt.tight_layout()
        plt.savefig("{}/degree_dist_plot_{}_s_{}.pdf".format(path0, saving_name, name0))
        plt.clf()
        with open(
            "{}/degree_dist_plot_data_{}_s_{}.npy".format(path0, saving_name, name0),
            "wb",
        ) as f0:
            np.save(f0, counts0)
            np.save(f0, bines0)


def cumulative_degree_distribution_plot(
    G0, type0=0, bins0=20, name0="Default sample name", path0="default_path"
):
    """
    Function that receives a directed network, the type of degree (0 for all nodes, 1 for in
    nodes and 2 for out nodes and the number of bins (20 by default) and computes distribution
    of the degrees and plots it
    """
    try:
        assert (
            type0 == 0 or type0 == 1 or type0 == 2
        )  # Tests that the clustering degree is in fact a number between 0 and 1
    except AssertionError:
        print(
            "Input type should be either 0 for all nodes, 1 for in nodes or 2 for out nodes"
        )
    else:
        if type0 == 0:
            x0 = np.asarray(G0.degree())[:, 0]
            y0 = np.asarray(G0.degree())[:, 1]
            plt.xlabel("Degree (d)")
            nametype = "all nodes"
            saving_name = "all_nodes"
        elif type0 == 1:
            x0 = np.asarray(G0.in_degree())[:, 0]
            y0 = np.asarray(G0.in_degree())[:, 1]
            plt.xlabel(r"Indegree (d$^-$)")
            nametype = "in degree nodes"
            saving_name = "in_nodes"
        else:
            x0 = np.asarray(G0.out_degree())[:, 0]
            y0 = np.asarray(G0.out_degree())[:, 1]
            plt.xlabel(r"Outdegree (d$^+$)")
            nametype = "out degree nodes"
            saving_name = "out_nodes"
        values0, base0 = np.histogram(y0, bins=40)
        cumulative0 = np.cumsum(values0)
        # plot the cumulative
        plt.plot(base0[:-1], cumulative0, c="blue", label="Cumulative dist.")
        # plot the survival function
        plt.plot(
            base0[:-1],
            len(y0) - cumulative0,
            c="green",
            label="Inverse cumulative dist.",
        )
        plt.yscale("log")
        plt.ylabel("Frequency")
        plt.legend()
        plt.title(
            "Cumulative degree distrbution plot for sample {} for {}".format(
                name0, nametype
            )
        )
        plt.tight_layout()
        plt.savefig(
            "{}/cum_degree_dist_plot_{}_s_{}.pdf".format(path0, saving_name, name0)
        )
        plt.clf()
        with open(
            "{}/cum_dist_plot_data_{}_s_{}.npy".format(path0, saving_name, name0), "wb"
        ) as f0:
            np.save(f0, base0[:-1])
            np.save(f0, cumulative0)
            np.save(f0, len(y0) - cumulative0)


def lorentz_curve_plot(G0, type0=0, name0="Default sample name", path0="default_path"):
    """
    Function that receives a directed network and the type of degree (0 for all nodes, 1 for in
    nodes and 2 for out nodes and computes the corresponding
    lorentz curve and plots it
    """
    try:
        assert (
            type0 == 0 or type0 == 1 or type0 == 2
        )  # Tests that the clustering degree is in fact a number between 0 and 1
    except AssertionError:
        print(
            "Input type should be either 0 for all nodes, 1 for in nodes or 2 for out nodes"
        )
    else:
        if type0 == 0:
            x0 = np.asarray(G0.degree())[:, 0]
            y0 = np.asarray(G0.degree())[:, 1]
            plt.xlabel("Degree (d)")
            info = "Degree"
            nametype = "all nodes"
            saving_name = "all_nodes"
        elif type0 == 1:
            x0 = np.asarray(G0.in_degree())[:, 0]
            y0 = np.asarray(G0.in_degree())[:, 1]
            plt.xlabel(r"Indegree (d$^-$)")
            info = r"Indegree (d$^-$)"
            nametype = "in degree nodes"
            saving_name = "in_nodes"
        else:
            x0 = np.asarray(G0.out_degree())[:, 0]
            y0 = np.asarray(G0.out_degree())[:, 1]
            plt.xlabel(r"Outdegree (d$^+$)")
            info = r"Outdegree (d$^+$)"
            nametype = "out degree nodes"
            saving_name = "out_nodes"
        y0 = np.sort(y0)
        sume = np.sum(y0)
        y0 = y0 / sume
        Y0 = np.cumsum(y0)
        x00 = (np.arange(np.size(y0)) + 1.0) / np.size(y0)
        lorentz_area = np.trapz(Y0, x=x00)
        total_area = np.trapz(x00, x=x00)
        Gi = np.round((total_area - lorentz_area) / total_area, 2)
        plt.plot(x00, Y0, label=info + " with G= {}".format(Gi))
        plt.plot(x00, x00, c="black", linestyle="--")
        plt.plot(x00, 1.0 - x00, c="black", linestyle="--")
        plt.fill_between(x00, x00, Y0, color="lightgray")
        plt.ylabel("Portion of edges")
        plt.xlabel("Portion of nodes with smallest degrees")
        plt.legend()
        plt.title("Lorentz curve plot for sample {} for {}".format(name0, nametype))
        plt.tight_layout()
        plt.savefig(
            "{}/lorentz_curve_plot_{}_s_{}.pdf".format(path0, saving_name, name0)
        )
        plt.clf()
        with open(
            "{}/lorentz_curve_plot_data_{}_s_{}.npy".format(path0, saving_name, name0),
            "wb",
        ) as f0:
            np.save(f0, x00)
            np.save(f0, Y0)
            np.save(f0, 1.0 - x00)


def clustering_degree(G0, p0):
    """
    Function that receives a directed network and a clustering degree p0 which is a number from 0 to 1 which will be rounded to two decimals
    and computes the probability that a node has clustering degree p0
    """
    try:
        assert (
            p0 <= 1.0 and p0 >= 0.0
        )  # Tests that the clustering degree is in fact a number between 0 and 1
    except AssertionError:
        print("Input clustering degree is not a number between 0 and 1")
    else:
        ni = np.fromiter(
            nx.clustering(G0).keys(), dtype=int
        )  # numpy array containing the clustering probability id of the whole network with only 2 decimals
        ci = np.round(
            np.fromiter(nx.clustering(G0).values(), dtype=float), 2
        )  # numpy array containing the clustering probability of the whole network with only 2 decimals
        n0 = G0.number_of_nodes()  # Number of nodes in the network G0
        n_c0 = ci[
            ci == p0
        ]  # numpy array containing the amount of nodes with clustering p0
        ni_c0 = ni[ci == p0]  # numpy array containing the id of nodes with degree k0
    return np.size(ni_c0) / n0


def clustering_degree_plot(
    G0, bins0=20, name0="Default sample name", path0="default_path"
):
    """
    Function that receives a directed network and the number of bins (default=20) and plots the clustering probability distribution
    in log scale in the y axis
    """
    ni = np.fromiter(
        nx.clustering(G0).keys(), dtype=int
    )  # numpy array containing the clustering probability id of the whole network
    ci = np.fromiter(
        nx.clustering(G0).values(), dtype=float
    )  # numpy array containing the clustering probability of the whole network
    n0 = G0.number_of_nodes()  # Number of nodes in the network G0
    with np.errstate(divide="ignore"):
        counts0, bines0, nn0 = plt.hist(
            np.log10(ci),
            range=(
                np.nanmin(np.log10(ci)[np.log10(ci) != -np.inf]),
                np.max(np.log10(ci)),
            ),
        )
    plt.yscale("log")
    plt.ylabel(r"Frequency $\log_{10} p(c)$")
    plt.xlabel(r"$\log_{10} c $")
    plt.title("Clustering degree distribution plot for sample {}".format(name0))
    plt.tight_layout()
    plt.savefig("{}/clust_dist_plot_s_{}.pdf".format(path0, name0))
    plt.clf()
    with open("{}/clust_dist_plot_data_s_{}.npy".format(path0, name0), "wb") as f0:
        np.save(f0, counts0)
        np.save(f0, bines0)


def betweenness_centrality_plot(
    G0, bins0=20, name0="Default sample name", path0="default_path"
):
    """
    Function that receives a directed network and the number of bins (default=20) and plots the betweenness centrality probability distribution
    in log scale in the y axis
    """
    ni = np.fromiter(
        nx.betweenness_centrality(G0).keys(), dtype=int
    )  # numpy array containing the betweenness centrality probability id of the whole network
    bi = np.fromiter(
        nx.betweenness_centrality(G0).values(), dtype=float
    )  # numpy array containing the betweenness centrality probability of the whole network
    n0 = G0.number_of_nodes()  # Number of nodes in the network G0
    # with np.errstate(divide='ignore'):
    #    plt.hist(np.log10(bi),range=(np.nanmin(np.log10(bi)[np.log10(bi) != -np.inf]),np.max(np.log10(bi))))
    counts0, bines0, nn0 = plt.hist(bi)
    plt.yscale("log")
    # plt.xscale("log")
    # plt.ylabel(r"Frequency $\log_{10} p(b)$")
    # plt.xlabel(r"$\log_{10} b$")
    plt.ylabel(r"Frequency $ p(b)$")
    plt.xlabel(r"$b$")
    plt.title("Betweenness centrality plot distribution for sample {}".format(name0))
    plt.tight_layout()
    plt.savefig("{}/betweennes_dist_plot_s_{}.pdf".format(path0, name0))
    plt.clf()
    with open("{}/betweennes_dist_plot_data_s_{}.npy".format(path0, name0), "wb") as f0:
        np.save(f0, counts0)
        np.save(f0, bines0)


def degree_correlation_plot(
    G0, source0="out", target0="out", name0="Default sample name", path0="default_path"
):
    import matplotlib as mpl

    """
    Function that receives a directed network, source and target node (in or out) for directed networks and plots the approximate degree-degree correlation
    matrix in log scale.
    """
    degrees = []
    neighbor_degrees = []
    x0 = np.asarray(G0.degree())[
        :, 0
    ]  #  numpy array containing the number of the node i am looking
    y0 = np.asarray(G0.degree())[:, 1]  #  numpy array containing the degree of the node
    ni = np.fromiter(
        nx.average_neighbor_degree(G0, source=source0, target=target0).keys(), dtype=int
    )  # numpy array containing the id of the node whom average_neighbor_degree is calculated
    bi = np.fromiter(
        nx.average_neighbor_degree(G0, source=source0, target=target0).values(),
        dtype=float,
    )  # numpy array containing the average_neighbor_degree of the whole network
    counts0, binesx0, binesy0, nn0 = plt.hist2d(y0, bi, norm=mpl.colors.LogNorm())
    # plt.yscale("log")
    plt.title(
        "Average degree-degree correlation plot for {} degree source and {} degree target for sample {}".format(
            source0, target0, name0
        )
    )
    plt.xlabel(r"$k$")
    plt.ylabel(r"$<k_{nn}(k)>$")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(
        "{}/degree_corr_plot_{}_{}_s_{}.pdf".format(path0, source0, target0, name0)
    )
    plt.clf()
    with open(
        "{}/degree_corr_plot_data_{}_{}_s_{}.npy".format(
            path0, source0, target0, name0
        ),
        "wb",
    ) as f0:
        np.save(f0, counts0)
        np.save(f0, binesx0)
        np.save(f0, binesy0)


def core_number(G):
    """Returns the core number for each vertex.

    A k-core is a maximal subgraph that contains nodes of degree k or more.

    The core number of a node is the largest value k of a k-core containing
    that node.

    Parameters
    ----------
    G : NetworkX graph
       A graph or directed graph

    Returns
    -------
    core_number : dictionary
       A dictionary keyed by node to the core number.

    Raises
    ------
    NetworkXError
        The k-core is not implemented for graphs with self loops
        or parallel edges.

    Notes
    -----
    Not implemented for graphs with parallel edges or self loops.

    For directed graphs the node degree is defined to be the
    in-degree + out-degree.

    References
    ----------
    .. [1] An O(m) Algorithm for Cores Decomposition of Networks
       Vladimir Batagelj and Matjaz Zaversnik, 2003.
       https://arxiv.org/abs/cs.DS/0310049
    """
    if nx.number_of_selfloops(G) > 0:
        msg = (
            "Input graph has self loops which is not permitted; "
            "Consider using G.remove_edges_from(nx.selfloop_edges(G))."
        )
        raise NetworkXError(msg)
    degrees = dict(G.degree())
    # Sort nodes by degree.
    nodes = sorted(degrees, key=degrees.get)
    bin_boundaries = [0]
    curr_degree = 0
    for i, v in enumerate(nodes):
        if degrees[v] > curr_degree:
            bin_boundaries.extend([i] * (degrees[v] - curr_degree))
            curr_degree = degrees[v]
    node_pos = {v: pos for pos, v in enumerate(nodes)}
    # The initial guess for the core number of a node is its degree.
    core = degrees
    nbrs = {v: list(nx.all_neighbors(G, v)) for v in G}
    for v in nodes:
        for u in nbrs[v]:
            if core[u] > core[v]:
                nbrs[u].remove(v)
                pos = node_pos[u]
                bin_start = bin_boundaries[core[u]]
                node_pos[u] = bin_start
                node_pos[nodes[bin_start]] = pos
                nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
                bin_boundaries[core[u]] += 1
                core[u] -= 1
    return core


def k_cores(G0, name0="Default sample name", path0="default_path"):
    """
    Function that receives a network G0 as argument and plots the histogram of the possible k cores as a function of their degree k
    """
    G0.remove_edges_from(nx.selfloop_edges(G0))
    kcors = core_number(G0)
    x0 = np.fromiter(kcors.keys(), dtype=float)
    y0 = np.fromiter(kcors.values(), dtype=float)
    print(np.max(y0))
    counts0, bines0, nn0 = plt.hist(y0, bins=int(np.max(y0) + 1))
    plt.title("K-cores as function of degree k plot for sample {}".format(name0))
    plt.yscale("log")
    plt.ylabel(r"Frequency $k_{core}(k)$")
    plt.xlabel(r"Degree $k$")
    plt.tight_layout()
    plt.savefig("{}/k_cores_plot_s_{}.pdf".format(path0, name0))
    plt.clf()
    with open("{}/k_cores_plot_data_s_{}.npy".format(path0, name0), "wb") as f0:
        np.save(f0, counts0)
        np.save(f0, bines0)
        np.save(f0, nn0)


def diameter(G0):
    """
    Function that receives a network G0 as argument and returns the lower bound of the diameter of graph G0
    """
    try:
        dia = nx.diameter(G0)
        print("Diameter of the network is: ", dia)
        return dia
    except:
        print("Network is not strongly connected, can't calculate diameter :(")
        pass


from collections.abc import MutableMapping
import pandas as pd


def flatten_dict(d: MutableMapping, sep: str = ".") -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient="records")
    #print(flat_dict)
    return flat_dict


def shortest_path(G0, name0="Default sample name", path0="default_path"):
    """
    Function that receives a network G0 as argument and plots the histogram of the shortest paths of all notes in the network G0
    """
    length = dict(nx.all_pairs_shortest_path_length(G0))
    length = flatten_dict(length)
    # length = nx.all_pairs_shortest_path_length(G0)
    vals0 = np.fromiter(length.values(), dtype=int)
    counts0, bines0, nn0 = plt.hist(vals0)
    plt.title("Shortest path probability plot for sample {}".format(name0))
    plt.xlabel(r"Shortest path $I$")
    plt.ylabel(r"Frequency $p(I)$")
    plt.tight_layout()
    plt.savefig("{}/shortest_path_plot_s_{}.pdf".format(path0, name0))
    plt.clf()
    with open("{}/shortest_path_plot_data_s_{}.npy".format(path0, name0), "wb") as f0:
        np.save(f0, counts0)
        np.save(f0, bines0)


def random_walk(G0, s0=100, d0=0.85, name0="Default sample name", path0="default_path"):
    """
    Function that receives a network G0 as argument and plots the probability of reaching
    each of the nodes starting from a random node using the pagerank algorithm
    with s0 maximum number of iterations in power method eigenvalue solver with default s0=100 and damping
    parameter d0=0.85 by deffault
    """
    import matplotlib as mpl
    from scipy.stats import gaussian_kde

    dats = np.array(list(nx.pagerank(G0, alpha=d0, max_iter=s0).items())).transpose()
    xy0 = np.vstack([dats[0], dats[1]])
    z0 = gaussian_kde(xy0)(xy0)
    idx0 = z0.argsort()
    x0, y0, z0 = dats[0][idx0], dats[1][idx0], z0[idx0]
    plt.scatter(x0, y0, c=z0, cmap="plasma")
    plt.title(
        r"Probability plot of reaching each node for"+ "\n"+r" sample {} with max iter {} and damping $d_0=${}".format(
            name0, s0, d0
        )
    )
    plt.ylabel(r"Probability of reachability $p_r(n)$")
    plt.xlabel(r"Node $n$")
    plt.yscale("log")
    plt.colorbar(label=r"Points density")
    plt.tight_layout()
    plt.savefig("{}/random_walk_plot_s_{}_s0_{}_d0_{}.pdf".format(path0, name0, s0, d0))
    plt.clf()
    with open(
        "{}/random_walk_plot_data_s_{}_s0_{}_d0_{}.npy".format(path0, name0, s0, d0),
        "wb",
    ) as f0:
        np.save(f0, x0)
        np.save(f0, y0)
        np.save(f0, z0)
