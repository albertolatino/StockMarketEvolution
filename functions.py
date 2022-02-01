# Importing libraries
import datetime
import pandas_datareader as pdr
import pandas as pd
from pandas import *
import numpy as np
import scipy.spatial
from scipy.spatial.distance import cdist
import requests
from igraph import *
import igraph
import io
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from ast import literal_eval
from scipy.stats import linregress
from scipy.spatial.distance import pdist, squareform


# This function gets a csv containing companies of the Index S&P 500, including their name, ticker and sector.
def get_companies():
    url = "https://pkgstore.datahub.io/core/s-and-p-500-companies/constituents_csv/data/1beb8c524488247ccb27917bfcb581ec/constituents_csv.csv"
    s = requests.get(url).content
    companies = pd.read_csv(io.StringIO(s.decode('utf-8')))
    return companies


# This function returns the log prices of the adjusted returns.
# Since the log for the day 1 is Nan, the first row is removed
def get_data(start: datetime, end: datetime, tickers: list):
    data = pdr.get_data_yahoo(tickers, start, end)
    data = data['Adj Close']
    log_returns = np.log(data / data.shift())
    log_returns = log_returns.iloc[1:, :]
    return log_returns


# This function reads an existing csv or downloads and store a new one
def read_data(start: datetime, end: datetime, tickers: list, newData: bool):
    if newData:
        data = get_data(start, end, tickers)
        data.to_csv(path_or_buf='S&P500_' + str(start.date()) + '_' + str(end.date()))

    else:
        data = pd.read_csv(filepath_or_buffer='S&P500_' + str(start.date()) + '_' + str(end.date()))

    return data


# Given an historical dataset, it returns data in the time range [start-emd]
def filter_by_date(data: DataFrame, start: datetime, end: datetime):
    filtered_data = data[(data['Date'] >= start) & (data['Date'] <= end)]
    return filtered_data


# Given a correlation dataframe, this function create a network.
# Each stock is a node of the network and edges are added in case the correlation between
# two stocks is greater than the threshold passed as argument
# Returns the created network
def create_network(tickers: list, sectors: list, correlations: DataFrame, threshold: int):
    color_dict = {"Industrials": "orange", "Information Technology": "red", "Financials": "blue", "Health Care": "pink",
                  "Consumer Discretionary": "yellow",
                  "Consumer Staples": "white", "Real Estate": "cyan", "Utilities": "gray", "Materials": "purple",
                  "Communication Services": "magenta", "Energy": "green"}
    # Assign tickers and sectors to nodes
    g = Graph(n=len(tickers))
    g.vs['name'] = tickers
    g.vs['sector'] = sectors
    g.vs["label"] = g.vs["name"]
    g.vs["label_size"] = 6
    g.vs["color"] = [color_dict[sector] for sector in g.vs["sector"]]

    # Iterate over the correlation dataframe and add edges
    for ticker1 in tickers:
        for ticker2 in tickers:
            id1 = g.vs.find(name=ticker1)
            id2 = g.vs.find(name=ticker2)
            if not g.are_connected(id1, id2) and ticker1 != ticker2 and correlations[ticker1][ticker2] >= threshold:
                g.add_edge(id1, id2)
    return g


# This function takes as argument a list of clusters ordered by size and
# computes percentages of sectors in the biggest n clusters. Parameter n is passed
# As num_clusters to the function. Returns a dictionary containing percentages of sectors for each cluster
def clustering_result(clusters: list, num_clusters: int):
    clusters_sector = []
    count = 0

    for j in range(0, num_clusters):

        if num_clusters == 1:
            cluster = clusters
        else:
            cluster = clusters[j]

        count_dict = {"Industrials": 0, "Information Technology": 0, "Financials": 0, "Health Care": 0,
                      "Consumer Discretionary": 0, "Consumer Staples": 0, "Real Estate": 0, "Utilities": 0,
                      "Materials": 0,
                      "Communication Services": 0, "Energy": 0}

        perc_dict = {"Industrials": 0, "Information Technology": 0, "Financials": 0, "Health Care": 0,
                     "Consumer Discretionary": 0, "Consumer Staples": 0, "Real Estate": 0, "Utilities": 0,
                     "Materials": 0,
                     "Communication Services": 0, "Energy": 0}

        for node in cluster.vs.indices:
            count_dict[cluster.vs[node]["sector"]] = count_dict[cluster.vs[node]["sector"]] + 1
            count = count + 1

        c = Counter(count_dict)
        most_common = c.most_common(3)
        keys = [key for key, val in most_common]
        values = [val for key, val in most_common]
        percentages = []

        index = 0
        for count in values:
            percentages.append(count / cluster.vcount())
            perc_dict[keys[index]] = count / cluster.vcount()
            index += 1

        # for i in range(0, 3):
        # print(keys[i] + "__" + str(round(percentages[i] * 100, 2)) + "%")

        # fig = plt.figure()
        # fig.suptitle('CLUSTER ' + str(j), fontsize=20)
        # plt.ylim(0, 1.0)
        # # plt.xticks(rotation='vertical')
        # plt.tight_layout()
        # plt.bar(keys, percentages, width=0.4)
        # plt.show()

        clusters_sector.append(perc_dict)

    return clusters_sector


# This function plots mean correlation over time for each one of the proposed window sized
def compute_optimal_window(filtered_data: DataFrame):
    sizes = [14, 28, 56, 112]

    for size in sizes:
        print("Size", size)
        x = []
        y = []
        # T is incremented by 'size' at each iteration
        for t in range(1, len(filtered_data.index), size):
            if t + size > len(filtered_data.index):
                break
            # Computes correlation of stocks in that window of time
            corr = filtered_data.iloc[t:t + size].corr(method=distcorr)
            mean_corr = corr.values[np.triu_indices_from(corr.values, 1)].mean()
            x.append(t)
            y.append(mean_corr)

        # Plot results
        plt.plot(x, y)
        plt.xlabel('Time')
        plt.ylabel('Mean Correlation')
        plt.legend(sizes, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

    plt.show()


# Once the window length has been chosen, this function
# makes a plot in order to analyse the behavior of different overalapping days values

def compute_best_overlapping(data: DataFrame, window_length: int):
    sizes = [14, 28, 42]
    for sliding_size in sizes:
        x = []
        y = []

        # Iterate through the historical data
        for t in range(1, len(data.index), sliding_size):
            if t + window_length > len(data.index):
                break

            corr = data.iloc[t:t + window_length].corr(method=distcorr)
            mean_corr = corr.values[np.triu_indices_from(corr.values, 1)].mean()
            x.append(t)
            y.append(mean_corr)

        plt.plot(x, y)
    plt.xlabel('Time')
    plt.ylabel('Mean Correlation')
    plt.legend(sizes, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

    plt.show()


# This class gathers all the necessary inforamtion to perform analysis and clustering through time
class Snapshot:
    def __init__(self, start, end, threshold, mean_correlation, network, clusters):
        self.start = start
        self.end = end
        self.threshold = threshold
        self.mean_correlation = mean_correlation
        self.network = network
        self.clusters = clusters


# This function takes as argument the temporal evolution of the stock market
# For the 'clusters_num' biggest clusters in each snapshot, computes the
# 'influential_num' most influent nodes in each cluster. Influence is measured in terms of
# eigenvector_centrality.
def rank_central_nodes(evolution: Snapshot, clusters_num: int, influential_num: int):
    results = []
    clusters = evolution.clusters
    clusters.sort(key=lambda x: x.vcount(), reverse=True)

    for i in range(0, clusters_num):
        subgraph = clusters[i]
        centralities = subgraph.eigenvector_centrality()

        # This are the indexes that order the nodes by centrality
        indexes = np.argsort(centralities)

        print(
            "Most Influent stocks - Cluster " + str(i) + ",   Size: " + str(
                subgraph.vcount()) + " :")

        if subgraph.vcount() < influential_num:
            influential_num = subgraph.vcount()

        # For each cluster take the first influential_num stocks
        for j in range(1, influential_num + 1):
            max_idx = indexes[-j]
            name = subgraph.vs[max_idx]['name']
            sector = subgraph.vs[max_idx]['sector']
            centrality = centralities[max_idx]

            print(name + ' --', sector + ", Centrality:", centrality)

            results.append([evolution.start, evolution.end, i, subgraph.vcount(), name, sector, centrality])

        print("\n")

    return results


# Given a dataset, a window length and an overlapping parameter
# It computes the networks over time and for each one apply clustering
# Thresholds for each network are chosen dynamically based on the mean correlation of each time window
# Nodes belonging to the same cluster saved with the same color
def temporal_clustering(filtered_data: DataFrame, window_length: int, tickers: list, sectors: list,
                        sliding_size: int):
    evolution = []

    if sliding_size > window_length:
        print("Overlappping days cannot be greater than window length ")

    # Iterate through the historical data
    for t in range(1, len(filtered_data.index), sliding_size):

        if t + window_length >= len(filtered_data.index):
            print("WINDOW SIZE ERROR")
            break

        corr = filtered_data.iloc[t:t + window_length].corr(method=distcorr)
        mean_corr = corr.values[np.triu_indices_from(corr.values, 1)].mean()
        # print("MEAN CORRELATION FOR THIS WINDOW:", mean_corr)

        # Set threshold  for the current window
        thr = 1.7 * mean_corr

        # Set an upperbound to the threshold.
        # Network with high thresholds resulted to be too sparse
        if thr >= 0.80:
            thr = 0.80

        print("MEAN CORRELATION FOR THIS WINDOW")
        print(mean_corr)

        g = create_network(tickers, sectors, correlations=corr, threshold=thr)
        clusters = g.community_multilevel()
        modularity = g.modularity(clusters.membership)
        pal = igraph.drawing.colors.ClusterColoringPalette(len(clusters))
        g.vs['color'] = pal.get_many(clusters.membership)
        start = filtered_data.iloc[[t]]["Date"].values
        end = filtered_data.iloc[[t + window_length]]["Date"].values
        print("PERIOD OF TIME:" + str(start) + "___" + str(end))
        print('Modularity for threshold ' + str(thr) + '= ' + str(modularity))
        igraph.plot(g, "cluster_from_" + str(start) + "_to_" + str(end) + ".png")
        # clustering_result(clusters, g)
        cluster_graphs = clusters.subgraphs()
        cluster_graphs.sort(key=lambda x: x.vcount(

        ), reverse=True)
        s = Snapshot(start, end, thr, mean_corr, g, cluster_graphs)
        evolution.append(s)

    return evolution


# Compute distance correlation as explained in the report.
def distcorr(X, Y):
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor


# Implementation of split penalty used for experiments in modularity optimization
def split_penalty(clusters: VertexClustering, g: Graph):
    i = len(clusters)
    memberships = clusters.membership
    edges_from_i_to_j = np.zeros([i, i])
    results = []

    for edge in g.es:
        source_vertex_id = edge.source
        target_vertex_id = edge.target
        source_cluster = memberships[source_vertex_id]
        target_cluster = memberships[target_vertex_id]

        if source_cluster != target_cluster:
            edges_from_i_to_j[source_cluster][target_cluster] += 1

    for cluster in range(1, len(clusters)):
        out_from_cluster = np.sum(edges_from_i_to_j[cluster])
        result = out_from_cluster / (2 * g.ecount())
        results.append(result)

    return sum(results)


# Compute Jaccard similarity of the two graphs passed as argument
# c1 and c2 are two subgraphs corresponding to clusters
def jaccard_similarity(c1: Graph, c2: Graph):
    c1_stocks = c1.vs()['name']
    c2_stocks = c2.vs()['name']
    intersection = len(list(set(c1_stocks) & set(c2_stocks)))
    union = (len(c1_stocks) + len(c2_stocks)) - intersection
    return float(intersection) / union


# Given two sets of clusters coming from two different snapshots
# It associate clusters with one another by looking at associations
# that maximize jaccard similarity
# Returns clusters coupled with each other
def track_clusters(subgraphs_t1: list, subgraphs_t2: list):
    x = len(subgraphs_t1)
    y = len(subgraphs_t2)

    # Matrix that for each one of the clusters in t1 says the jaccard score w.r.t. the clusters in t2
    similarity_matrix = [[0 for j in range(y)] for i in range(x)]
    best_matches = []

    for index_t1 in range(0, x):
        max = 0
        best_match_index = 0
        for index_t2 in range(0, y):
            similarity_matrix[index_t1][index_t2] = jaccard_similarity(subgraphs_t1[index_t1], subgraphs_t2[index_t2])
            if similarity_matrix[index_t1][index_t2] > max:
                max = similarity_matrix[index_t1][index_t2]
                best_match_index = index_t2
        best_matches.append(best_match_index)

    return best_matches


# Given a network, computes the degree sequence of all the nodes in the network
# Returns an histogram-like counting the occurrence of degree values in the network
def degree_distribution(g: Graph):
    degs = {}
    for n in g.vs.indices:
        deg = g.vs[n].degree()
        if deg not in degs.keys():
            degs[deg] = 0
        degs[deg] += 1
    items = sorted(degs.items())
    x = [k for (k, v) in items]
    y = [v for (k, v) in items]

    return x, y


# Given the sequence of snapshots , this function computes several metrics and returns:
#
# df: for each snapshot contains several information-- Start', 'End', 'Window', 'Avg Corr', 'Thr', 'Edges',
# 'Communities', 'Clustering Coeff','Avg Degree', 'Avg Shortest Path', 'X', 'Y'
#
# df_stocks, returns for each window measures regarding the importance of each stock --
# Start', 'End', 'Window', 'Ticker', 'Sector', 'Degree', 'Betweenness','Centrality'
#
# df_clusters, returns information regarding clusters and the most important nodes inside of clusters
# -- 'Start', 'End', 'Cluster', 'Size', 'Name', 'Sector', 'Centrality'
def evolution_results(evolution: list):
    results = []
    stocks_results = []
    cluster_results = []
    for t in range(0, len(evolution)):
        # Create data describing the time window
        g = evolution[t].network
        window = t
        start = evolution[t].start
        end = evolution[t].end
        mean_corr = evolution[t].mean_correlation
        thr = evolution[t].threshold
        num_edges = g.ecount()
        num_of_communities = sum(cluster.vcount() > 4 for cluster in evolution[t].clusters)
        avg_clustering = g.transitivity_undirected()
        avg_degree = np.mean(g.vs.degree())
        avg_shortest_path = np.mean(g.shortest_paths())

        # Compute degreee distribution for that snapshot
        x, y = degree_distribution(g)
        results.append(
            [start, end, window, mean_corr, thr, num_edges, num_of_communities, avg_clustering,  # betweenness,
             avg_degree, avg_shortest_path, x, y])

        centralities = g.eigenvector_centrality()

        # Iterates over the stock of each snapshot
        for node in g.vs.indices:
            ticker = g.vs[node]['name']
            degree = g.vs[node].degree()
            betweenness = round(g.betweenness(vertices=g.vs[node], directed=False), 1)
            centrality = centralities[node]
            sector = g.vs[node]['sector']
            stocks_results.append([start, end, window, ticker, sector, degree, betweenness, centrality])

        # Most important nodes in clusters
        cluster_result = rank_central_nodes(evolution[t], clusters_num=5, influential_num=4)
        cluster_results.extend(cluster_result)

    df_stocks = pd.DataFrame(stocks_results,
                             columns=['Start', 'End', 'Window', 'Ticker', 'Sector', 'Degree', 'Betweenness',
                                      'Centrality'])

    df_clusters = pd.DataFrame(cluster_results,
                               columns=['Start', 'End', 'Cluster', 'Size', 'Name', 'Sector', 'Centrality'])

    df = pd.DataFrame(results,
                      columns=['Start', 'End', 'Window', 'Avg Corr', 'Thr', 'Edges', 'Communities', 'Clustering Coeff',
                               'Avg Degree', 'Avg Shortest Path', 'X', 'Y'])

    return df, df_stocks, df_clusters


# Once clusters are associated with each other through time
# The evolution of clusters'size is plotted over time
def clusters_plot_size(evolution: list, num_clusters: int, track_indexes: list):
    for i in range(0, num_clusters):
        sizes = []
        for t in range(0, len(evolution)):
            size = evolution[t].clusters[track_indexes[t][i]].vcount()
            sizes.append(size)
        plt.plot([i for i in range(0, len(evolution))], sizes, label="Cluster " + str(i))

    plt.legend(loc="upper right")
    plt.show()


# Plots the degrees of the network in a lok-log scale
def degree_plot(df: DataFrame):
    fig, ax = plt.subplots(nrows=4, ncols=6, figsize=(30, 30))
    t = 0  # Be sure that it plots all the dataframe
    for row in ax:
        for col in row:
            x = literal_eval(df['X'].iloc[t])
            x = np.array(x)
            y = literal_eval(df['Y'].iloc[t])
            col.semilogx()
            col.semilogy()
            col.scatter(x, y, color='blue')
            col.set_title(" Window from: " + str(df['Start'].iloc[t]) + " to" + str(df['End'].iloc[t]))
            t = t + 1

    plt.tight_layout()
    plt.show()


# Given a dataframe, plot the evolution of a certain measure over time
def plot_data(df: DataFrame, attribute: str):
    import matplotlib.ticker as plticker
    loc = plticker.MultipleLocator(base=5.0)  # this locator puts ticks at regular intervals
    plt.xlabel('Days')
    plt.ylabel(attribute)
    plt.xticks(rotation=75)
    ax = plt.gca()
    ax.xaxis.set_major_locator(loc)
    plt.tight_layout(pad=5)

    plt.plot(df['Start'], df[attribute])
    plt.show()


# Plot evolution of sectors in each tracked cluster
def plot_clusters_sectors(num_clusters: int, evolution: list, track_indexes_T: list):
    clusters_composition = []

    for cluster in range(0, num_clusters):
        for t in range(0, len(evolution)):
            result = clustering_result(evolution[t].clusters[track_indexes_T[cluster][t]], num_clusters=1)
            list = [t, cluster, track_indexes_T[cluster][t]]
            list.extend(result[0].values())
            clusters_composition.append(list)

            fig = plt.figure()
            fig.suptitle(
                'CLUSTER ' + str(cluster) + '- match cluster- ' + str(track_indexes_T[cluster][t]) + ', TIME = ' + str(
                    t),
                fontsize=10)
            plt.ylim(0, 1.0)
            plt.xticks(rotation=65)
            plt.tight_layout(pad=6)
            plt.bar(result[0].keys(), result[0].values(), width=0.25)
            plt.show()

    # Save dataframe with sectors of clusters over time
    df_cluster_composition = pd.DataFrame(clusters_composition,
                                          columns=['Window', 'Cluster_t0', 'Cluster_t', 'Industrials',
                                                   'Information Technology',
                                                   'Financials', 'Health Care', 'Consumer Discretionary',
                                                   'Consumer Staples',
                                                   'Real Estate', 'Utilities', 'Materials', 'Communication Services',
                                                   'Energy'])

    return df_cluster_composition
