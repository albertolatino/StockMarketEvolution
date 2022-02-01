# Importing Libraries
from functions import *
import matplotlib.pyplot as plt
import pandas as pd
import igraph
from igraph import *
import time
import numpy as np
import json
from modularitydensity.metrics import modularity_density
import community

# Variables to control download of new dataset and
# To control Temporal Clustering
newData = False
temporalClustering = True

number_of_stocks = 505

# Get data about companies and select the amount of them to be analyzed
companies = get_companies()
analyzed_companies = companies.iloc[:number_of_stocks]
# Get time series data and compute distance correlation
tickers = analyzed_companies['Symbol'].tolist()

# Set interval of time to donwload the dataset
start = datetime.datetime(2012, 1, 1)
end = datetime.datetime(2021, 12, 31)
# Read dataset
data = read_data(start, end, tickers, newData)

# Clean data, less than ten stocks had some Nan values for some days
# due to market holidays. Since they were only few stocks, they were dropped.
# This problem could be solved by applying some interpolation method
data = data.dropna(axis='columns')
tickers = data.columns.values.tolist()
tickers.pop(0)
sectors = analyzed_companies[analyzed_companies['Symbol'].isin(tickers)]['Sector'].tolist()

# Chose analysed interval of time
d1 = '2016-01-01'
d2 = '2022-01-20'
filtered_data = filter_by_date(data, d1, d2)

# Compute correlation matrix
corr = filtered_data.corr(method=distcorr)
print("A")

# Number of clusters to monitor over time
num_clusters = 8

# This operation requires a lot of time, since uses also small time window for analysis
# compute_best_overlapping(filtered_data, window_length=56)
# This operation requires a lot of time, since uses also small time window for analysis
# Comment it to speed up
# compute_optimal_window(filtered_data)

# Optimal computed time window
window_length = 56
# Overlapping parameter
sld = 28

# List of snapshots
evolution = []

if temporalClustering:
    evolution = temporal_clustering(filtered_data, window_length, tickers, sectors, sliding_size=sld)

    # For each Cluster, returns the indexes of other clusters over time that match it the most
    track_indexes = [[0 for i in range(0, num_clusters)] for t in range(0, len(evolution))]

    for t in range(0, len(evolution)):
        print("from ", evolution[t].start + " to ", evolution[t].end)

        # Track cluster indexes over time
        indexes = track_clusters(subgraphs_t1=evolution[0].clusters[:num_clusters],
                                 subgraphs_t2=evolution[t].clusters[:num_clusters])

        for i in range(0, len(indexes)):
            print("SNAPSHOT ", t)
            print("Cluster: ", str(i) + "Matches cluster ", str(indexes[i]))
            # For each cluster this matrix records the indexes of its associated clusters
            track_indexes[t][i] = indexes[i]

    # Save information in dataframes
    df, df_stocks, df_clusters = evolution_results(evolution)
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    df.to_csv(path_or_buf='S&P500_Results_' + str(df.iloc[0]["Start"]) + '_' + str(df.iloc[0]["End"]))
    df_stocks.to_csv(path_or_buf='S&P500_Stocks_' + str(df.iloc[0]["Start"]) + '_' + str(df.iloc[0]["End"]))
    df.to_csv(path_or_buf='S&P500_Clusters_' + str(df.iloc[0]["Start"]) + '_' + str(df.iloc[0]["End"]))

# Read information from previously stored dataframes
df = pd.read_csv(filepath_or_buffer='S&P500_Results')
df_stocks = pd.read_csv(filepath_or_buffer='S&P500_Stocks')
df_clusters = pd.read_csv(filepath_or_buffer='S&P500_Clusters')
df_clusters_composition = pd.read_csv(filepath_or_buffer='S&P500_ClusterComposition')

# Plot evolving size of clusters
clusters_plot_size(evolution, num_clusters, track_indexes)
# Plot degree distribution
degree_plot(df)
# Plot metrics over time specified by 'attribute'
plot_data(df, attribute="Avg Degree")
plot_data(df, attribute="Communities")
plot_data(df, attribute="Avg Shortest Path")
plot_data(df, attribute="Clustering Coeff")
plot_data(df, attribute="Avg Corr")
plot_data(df, attribute="Thr")

# Plot influent nodes
df_stocks.sort_values(['Start', 'Betweenness'], ascending=False).groupby(['Start']).head(5).groupby(['Sector']).count()[
    'Ticker'].plot(kind='bar', color='blue', position=0, width=0.3)
df_stocks.sort_values(['Start', 'Degree'], ascending=False).groupby(['Start']).head(5).groupby(['Sector']).count()[
    'Ticker'].plot(kind='bar', color='red', position=1, width=0.3)
df_stocks.sort_values(['Start', 'Centrality'], ascending=False).groupby(['Start']).head(5).groupby(['Sector']).count()[
    'Ticker'].plot(kind='bar', color='green', position=2, width=0.3)
plt.legend(['Betweenness', 'Degree', 'Centrality'], loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
plt.xticks(rotation=80, fontsize=6)
plt.tight_layout(pad=4)
plt.show()

# Transpose for better interpretation of data
track_indexes_T = np.array(track_indexes).T.tolist()

#Information regarding the evolution of sectors in clusters over time
df_cluster_composition = plot_clusters_sectors(num_clusters,evolution, track_indexes_T)
df_cluster_composition.to_csv(path_or_buf='S&P500_ClusterComposition_')


