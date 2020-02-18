# Method
# 1.Centering
# 2.Standardization
# 3.Decorrelation
# 4.ReStandardization
# 5.Feature Reduction
# 6.Finding the optimal number of clusters
# 7.Deriving personas from clusters

import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets
import pickle

font = {'family': 'serif',
        'weight': 'bold',
        'size': 15}

matplotlib.rc('font', **font)
plt.style.use('fast')


def scale(data_to_scale):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(data_to_scale)
    data_normalized = scaler.transform(data_to_scale)
    print('Data is normalized')
    print('mean: ', data_normalized.mean(axis=0).round(2))
    print('std: ', data_normalized.std(axis=0).round(2))
    return data_normalized


def data_transformation(data, continues_cols):
    # Limit the number of columns for transformation
    data_log = data.copy(deep=True)

    for col in continues_cols:
        data_log.loc[:, col] = np.log(data_log.loc[:, col] + 1)
        data_normalized = scale(data_log)
    return data_normalized


def decorrelate_reduce(data, col_names):
    pca = PCA()
    decorr_data = pca.fit_transform(data)
    print(decorr_data.shape)
    print('Data is normalized')
    print('mean: ', decorr_data.mean(axis=0).round(2))
    print('std: ', decorr_data.std(axis=0).round(2))
    # get feature weights for each component plain
    weights = pd.DataFrame(pca.components_[0:2, ], columns=col_names[0:75], index=['PC-1', 'PC-2'])
    print('Weights for first two PCs %s' % weights)
    return decorr_data


def data_re_transformation(data):
    data_normalized = scale(data)
    return data_normalized


def visualize_distributions_for_skewness(data_vis):
    for col_vis in data_vis.columns:
        print(col_vis)
        try:
            sns_plot = sns.distplot(data_vis.loc[:, col_vis], bins=20)
            fig = sns_plot.get_figure()
            fig.savefig('figures/' + str(col_vis) + ".png")
            fig.clear()

            frequency_log = np.log(data_vis.loc[:, col_vis] + 1)
            log_plot_vis = sns.distplot(frequency_log)
            log_fig_vis = log_plot_vis.get_figure()
            log_fig_vis.savefig('figures/log' + str(col_vis) + ".png")
            plt.show()
        except:
            print("No data distribution, constant")


def visualize_pca_variance(features, data):
    # Check for pca variance
    pca = PCA()
    pca.fit_transform(data)

    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.xticks()
    plt.ylabel('variance ratio')
    plt.xlabel('PCA feature')
    plt.tight_layout()
    plt.savefig('figures/pca_variance_ratio.png')
    plt.show()

    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum())
    plt.xticks()
    plt.ylabel('cumulative sum of variances')
    plt.xlabel('PCA feature')
    plt.tight_layout()
    plt.savefig('figures/pca_variance_ratio_cumsum.png')
    plt.show()

    return pca


def visualize_feature_corr(data):
    corr_metrics = data.corr()
    corr_metrics.style.background_gradient()


def visualize_clusters_in_2d(data, num_components=2, k=3):
    # Visualize the results on PCA-reduced data

    reduced_data_high_dim = PCA().fit_transform(data)
    reduced_data = PCA(n_components=num_components).fit_transform(data)
    reduced_data = data_re_transformation(reduced_data)
    kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
    kmeans.fit(reduced_data)

    kmeans_real_clusters = KMeans(init='k-means++', n_clusters=k, n_init=10)
    kmeans_real_clusters.fit(reduced_data_high_dim)
    clusters = kmeans_real_clusters.predict(reduced_data_high_dim)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters)
    plt.legend(loc='upper right')

    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on Propser Applicants\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def visualize_clusters_in_2d_1(data, num_components=2, k=3):
    # Visualize the results on PCA-reduced data

    reduced_data = PCA(n_components=num_components).fit_transform(data)
    reduced_data = data_re_transformation(reduced_data)
    kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
    kmeans.fit(reduced_data)

    clusters = kmeans.predict(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters)
    plt.legend(loc='upper right')
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on Propser Applicants\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig('figures/2D.png')
    plt.show()


def visualize_elbow(data):
    # Fit KMeans and calculate SSE for each *k*
    sse = {}
    for k in range(1, 11):
        print('Running kmeans with k %s' % k)
        kmeans = KMeans(init='k-means++', n_clusters=k, random_state=10)
        kmeans.fit(data)
        sse[k] = kmeans.inertia_  # sum of squared distances to closest cluster center

    # Plot SSE for each *k*
    plt.figure()
    plt.title('The Elbow Method')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.tight_layout()
    sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
    plt.savefig('figures/elbow.png')


def cluster(data, original_data, k, r):
    # Run clustering 10 times, pick the one with min sse

    if r != -1:
        selected_kmeans = KMeans(n_clusters=k, random_state=r)
        selected_kmeans.fit(data)
    else:
        min_sse = 1000000000000
        for r in range(1, 20):
            print(r)
            kmeans = KMeans(n_clusters=k, random_state=r)
            kmeans.fit(data)
            sse = kmeans.inertia_
            if sse < min_sse:
                min_sse = sse
                selected_kmeans = kmeans
                selected_r = r
        print('selected r is %s' % selected_r)

    clusters = selected_kmeans.predict(data)
    original_data['clusters'] = clusters

    grouped_apps = original_data.groupby('clusters')
    centers = grouped_apps.mean()
    centers_median = grouped_apps.median()
    stds = grouped_apps.std()

    centers.to_csv('output/' + 'cluster_centers' + str(k) + str(r) + '.csv')
    stds.to_csv('output/' + 'cluster_stds' + str(k) + str(r) + '.csv')
    centers_median.to_csv('output/' + 'cluster_medians' + str(k) + str(r) + '.csv')

    pos_centers = grouped_apps.agg(lambda x: x[x > 0].mean())
    pos_centers_median = grouped_apps.agg(lambda x: x[x > 0].median())

    most_frequent_values = grouped_apps.agg(lambda x: x.value_counts().index[0])
    most_frequent_values.to_csv('output/' + 'frequent_values' + str(k) + '.csv')

    n_members_per_cluster = grouped_apps.count()
    print(n_members_per_cluster)

    print(grouped_apps['clusters'].count())

    return centers, clusters


def run_analysis(data, k):
    # Clean data
    # Already clean

    # Transform data to prepare for PCA/clustering
    transoformed_data = data_transformation(data, data.columns)

    # Decorrelate and reduce dimension with PCA
    decorrelated_data = decorrelate_reduce(transoformed_data, data.columns)

    re_standardized_pca_data = data_re_transformation(decorrelated_data)

    # Cluster data with k-means
    (cluster_centers, clusters) = cluster(re_standardized_pca_data, data, k, 1)
    print(cluster_centers)

    visualize_feature_corr(data)
    visualize_pca_variance(data.columns, transoformed_data)

    visualize_distributions_for_skewness(data)
    visualize_clusters_in_2d_1(transoformed_data, 2, k)

    visualize_elbow(data)


def main():

    # Load data here, try load_digits
    loaded_data = datasets.load_iris()
    data = pd.DataFrame(loaded_data.data)
    ground_truth = loaded_data.target
    k = 10

    # Create a directory named figures
    run_analysis(data, k)


if __name__ == '__main__':
    main()
