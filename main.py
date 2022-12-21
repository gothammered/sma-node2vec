import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from node2vec import Node2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

"""
기본 설정 파트
"""
# 열 출력 제한 수 조정
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 한글은 matplotlib 출력을 위해서 따로 설정이 필요하므로 font매니저 import
import matplotlib.font_manager

# 한글 출력을 위하여 font 설정
font_name = matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/HANDotum.ttf').get_name()
matplotlib.rc('font', family=font_name)


def tryNode2Vec(t_year, dimension_size):
    # Read data
    df = pd.read_csv('./data/transit_survey_SMA_{0}.csv'.format(t_year))
    adm_cd_list = df['ADM_CD_O'].unique().tolist()

    # Create OD matrix
    df_od_pivot = pd.pivot_table(df, index='ADM_CD_O', columns='ADM_CD_D', values='Unnamed: 0', aggfunc='count')

    # Create OD graph from data
    G = nx.Graph()
    for adm_cd_o in tqdm(adm_cd_list):
        # exclude 3135037 for year 2016 for being alone
        if t_year == '2016' and adm_cd_o == 3135037:
            continue
        for adm_cd_d in adm_cd_list:
            w = df_od_pivot.at[adm_cd_o, adm_cd_d]
            if np.isnan(w):
                continue
            else:
                G.add_edge(adm_cd_o, adm_cd_d, weight=w)

    # Print Graph info
    print('OD Graph of year {0}'.format(t_year))
    print('Node: {0}'.format(len(G)))
    print('Edge: {0}'.format(G.size()))
    print('Total number of commuter: {0}'.format(int(G.size(weight='weight'))))

    # Visualize the graph
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title('Commuting OD Graph of {0}'.format(t_year))
    nx.draw_networkx(G, ax=ax, node_size=15, linewidths=0.1)
    ax.set_xlabel('{0} nodes, {1} edges, {2} commuters'.format(len(G), G.size(), int(G.size(weight='weight'))))
    fig.savefig('./result/fig/ODGraph_{0}.png'.format(t_year), dpi=300)
    # plt.show()

    # Create Node2Vec model and learn
    n2v = Node2Vec(G, dimensions=dimension_size, walk_length=10, num_walks=100, p=1, q=2, workers=16)
    model = n2v.fit(window=10, min_count=1, batch_words=4)


    node_ids = model.wv.index_to_key
    model.wv.save_word2vec_format('./data/node2vec_result_{0}.csv'.format(t_year))
    node_embeddings = model.wv.vectors

    # Reduce dimension of vectors to 2 using t-SNE and save the result for further usage
    tsne = TSNE(n_components=2)
    node_embeddings_2d = tsne.fit_transform(node_embeddings)
    embedding_df = pd.DataFrame(node_embeddings_2d, index=node_ids, columns=['X', 'Y'])
    embedding_df.to_csv('./data/node2vec_result_2d_{0}.csv'.format(t_year))

    # Visualize the Word2Vec Result
    fig, ax = plt.subplots(figsize=(10, 8))
    color = {'11': 'b', '23': 'r', '31': 'g'}
    sido_nm = {'11': 'SEL', '23': 'ICN', '31': 'GYG'}
    for i, id in enumerate(node_ids):
        ax.scatter(node_embeddings_2d[i, 0], node_embeddings_2d[i, 1], c=color[id[0:2]], s=2, alpha=0.7, label=sido_nm[id[0:2]])
        ax.set_title('Node2Vec Result of {0}'.format(t_year))
    ax.legend(labels=['SEL', 'ICN', 'GYG'])
    fig.savefig('./result/fig/Node2Vec_TSNE_{0}.png'.format(t_year))
    # plt.show()

    # Run KMeans Clustering
    embedding_df = pd.DataFrame(node_embeddings, index=node_ids, columns=range(0, dimension_size))
    kmc = KMeans(n_clusters=8, init='random', max_iter=10000, random_state=0)
    kmc.fit(embedding_df)
    embedding_df['KMeans_8'] = kmc.labels_
    # Save the result for visualization
    embedding_df.to_csv('./result/node2vec_clustering_result_{0}.csv'.format(t_year))

    # Get mean vector for each cluster and find the most similar node to it
    similar_node_df = pd.DataFrame()
    for k in tqdm(range(0, 8)):
        cluster_df = embedding_df[embedding_df['KMeans_8'] == k]
        cluster_nodes = cluster_df.index.tolist()
        mean_vec = model.wv.get_mean_vector(cluster_nodes)
        similar_nodes = model.wv.similar_by_vector(mean_vec)
        for count, node in enumerate(similar_nodes):
            index_nm = 'cluster ' + str(k) + ' rank ' + str(count)
            similar_node_df.at[index_nm, 'cluster'] = k
            similar_node_df.at[index_nm, 'rank'] = count
            similar_node_df.at[index_nm, 'key'] = node[0]
            similar_node_df.at[index_nm, 'similarity'] = node[1]

    similar_node_df.to_csv('./result/node2vec_similarity_result_{0}.csv'.format(t_year))


# Get Inertia and Silhouette score to find the most optimal k
# The most optimal k was found to be 8
def runKMeansScore(t_year, dimension_size, max_clus, is2d=False):
    if is2d:
        embedding_df = pd.read_csv('./data/node2vec_result_2d_{0}.csv'.format(t_year))
        embedding_df = embedding_df.set_index('Unnamed: 0', drop=True)
    else:
        embedding_df = pd.read_csv('./data/node2vec_result_{0}.csv'.format(t_year), sep=' ',
                                   names=range(0, dimension_size))

    inertias = []
    silhouette = []

    for k in range(2, max_clus):
        kmc = KMeans(n_clusters=k, init='random', max_iter=1000, random_state=0)
        kmc.fit(embedding_df)
        inertias.append(kmc.inertia_)
        label_kmc = kmc.predict(embedding_df)
        silhouette.append(silhouette_score(embedding_df, label_kmc))

    fig, (ax_inertia, ax_silhouette) = plt.subplots(2, 1, figsize=(12, 12))
    ax_inertia.plot(range(2, max_clus), inertias, '-o', c='b', label='Inertia score')
    ax_silhouette.plot(range(2, max_clus), silhouette, '-x', c='r', label='Silhouette score')
    ax_inertia.set_title('Inertia score of {0}'.format(t_year))
    ax_inertia.set_xlabel('Number of clusters')
    ax_inertia.set_ylabel('Inertia score')
    ax_silhouette.set_title('Silhouette score of {0}'.format(t_year))
    ax_silhouette.set_xlabel('Number of clusters')
    ax_inertia.set_ylabel('Silhouette score')

    fig.suptitle('Inertia and Silhouette score of KMeans clustering, {0}'.format(t_year), fontsize=16)
    plt.show()


"""
RUN PART
"""
for t_year in ['2006', '2010', '2016']:
    # tryNode2Vec(t_year, 64)
    runKMeansScore(t_year, 64, 30, is2d=True)     # found 8 is the optimal k
