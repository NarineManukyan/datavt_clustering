import hdbscan
from sklearn import datasets
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering as Agglo
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy import stats
from sklearn.decomposition import PCA
from google.colab import drive
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
import altair as alt
from scipy.stats import mode
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
drive.mount('/content/gdrive')
my_drive = '/content/gdrive/My Drive/'

"""def cluster_evaluation(self, predicted): 
    predicted = np.array(self.cluster[predicted])
    predicted_index = np.argsort(predicted)
    actual = np.array(self.data[data.target])
    actual_index = np.argsort(actual)
    labels = []
    label = []
    clusters = []
    cluster = []

    # seperates the clusters into seperate lists, then retrieves their index
    for i in range(len(actual) - 1):
      a = predicted[predicted_index[i]]
      b = predicted[predicted_index[i + 1]]
      if i == (len(predicted)- 2):
        cluster.append(predicted_index[i])
        cluster.append(predicted_index[i + 1])
        a = 4
        b = 3
        clusters.append(cluster)
      elif a == b:
        cluster.append(predicted_index[i])
      else:
        cluster.append(predicted_index[i])
        clusters.append(cluster)
        cluster = []

    # seperates the labels into seperate lists, then retrieves their index
    for i in range(len(actual) - 1):
      a = actual[actual_index[i]]
      b = actual[actual_index[i + 1]]
      if i == (len(actual)- 2):
        label.append(actual_index[i])
        label.append(actual_index[i + 1])
        a = 4
        b = 3
        labels.append(label)
      elif a == b:
        label.append(actual_index[i])
      else:
        label.append(actual_index[i])
        labels.append(label)
        label = []

    # gets the size of the labels
    label_sizes = []
    for x in range(len(list(labels))):
      label_sizes.append(len(labels[x]))

    # gets the size of the clusers
    cluster_sizes = []
    for x in range(len(list(clusters))):
      cluster_sizes.append(len(clusters[x]))

    # makes the evaluation metric
    # find the predicted value of the the values that are in each cluster
    for i in range(len(clusters)):
      for x in range(len(clusters[i])):
        index = clusters[i][x]
        value = actual[index]
        clusters[i][x] = value
    
    # evaluation metric = (max/predicted n)/((label n - predicted n + 1)
    evaluations = []
    for i in range(len(clusters)):
      n = len(clusters[i])
      sort_cluster = sorted(clusters[i])
      
      # returns a list of the count of the cluster and the values
      cluster_group = list(Counter(sort_cluster).keys())
      cluster_value = list(Counter(sort_cluster).values())

      #runs through to find the dominate value of the cluster, and what value that cluster is
      for x in range(len(cluster_value)):
        max = cluster_value[0]
        max_group = cluster_group[0]
        others = 0
        if max < cluster_value[x]:
          others += max
          max = cluster_value[x]
          max_group = cluster_group[x]
        else:
          others += cluster_value[x]
      difference = (n - (label_sizes[max_group]) + 1)
      cluster_evaluation = [((max/n)/difference), max_group]
      evaluations.append(cluster_evaluation)
    return evaluations
"""


class Dataset():
    def __init__(self, data, target=''):
        self.data_original = data
        self.data = data
        self.clusters = {}
        self.categoricals = []
        self.numericals = []
        self.drops = []
        self.target = target
        self.missing_list = [' ', '', 'NO DATA']
        self.projection = None # for making TSNE easier
        self.label_strings = ['label', 'target', 'target_class', 'diagnosis']
    # TO DO: force object columns into string or float based on majority class.

    def assign_categories(self, categoricals=[], numericals=[], drops=[],
                          target=[]):
        self.categoricals = categoricals
        self.numericals = numericals
        self.drops = drops
        self.target = target

    def clean(self, find_categories=True, center=True, standardize=True, 
              pca=True, iter_impute=False, drop_thresh=1):
        # Drop threshold of 1.0 means no drops (except empty things)
        # ASSIGNING COLUMN TYPES

        self.data.replace(self.missing_list, np.nan, inplace=True)

        if find_categories:
            for column in self.data.columns:
                try:
                    self.data[column] = self.data[column].astype('float64')
                except ValueError:
                    if pd.to_numeric(self.data[column], errors='coerce').isna().sum() < .25 * len(self.data): #counting the numbers of NaNs
                        self.data = self.data[pd.to_numeric(self.data[column], errors='coerce').isna() ==  False]
                
                # to do: find the datatype of non-nan things in each column, go with majority. cut the rest. convert nan to '' if string.
                n_unique_values = len(np.unique(self.data[column]))
                if column in self.label_strings:
                    self.drops.append(column)
                elif self.data[column].dtype in ['float64', 'int64', 'float32', 'int8']:
                    spread = self.data[column].max() - self.data[column].min() + 1
                    if n_unique_values < 50 and spread == n_unique_values:
                        self.categoricals.append(column)
                    elif np.mean(self.data[column].sort_values().diff()) == 1:
                        self.drops.append(column)
                    else:
                        self.numericals.append(column)
                elif n_unique_values/len(self.data[column]) >= .75:
                    print(column, self.data[column].dtype)
                    self.drops.append(column)
                else:
                    self.categoricals.append(column)

        # ACTUAL PREPROCESSING
        self.data_original_categories = self.data.drop(columns=[])
        self.data = pd.get_dummies(self.data, columns=list(self.categoricals))
        self.data = self.data.drop(columns=self.drops)

        # PURGING AND FILLING MISSING DATA
        if not iter_impute:
            self.data.dropna(axis=1, thresh=int((1-drop_thresh)*len(self.data)), inplace=True) #COLUMNS
            self.data.dropna(axis=0, inplace=True) #ROWS
            self.data.index = pd.RangeIndex(len(self.data.index))

            for column in self.data.columns:
                if column in self.categoricals:
                    self.data[column] = self.data[column].apply(lambda x: mode(self.data[column], nan_policy='omit') if x==np.nan else x)
                else:
                    self.data[column] = self.data[column].apply(lambda x: np.nanmean(self.data[column]) if x==np.nan else x)

        if center:
            for each in self.numericals:
                signs = np.sign(self.data[each])
                self.data[each] = np.log(np.abs(self.data[each])+.1)
                self.data[each] = self.data[each] * signs
        if standardize:
            scaler = StandardScaler()
            self.data[self.data.columns] = scaler.fit_transform(self.data[self.data.columns])
        if iter_impute:
            imp_mean = IterativeImputer(random_state=0)
            print(imp_mean.fit_transform(self.data))
        if pca:
            pca_model = PCA()
            pca_model.fit(self.data[self.data.columns])
            for n in range(len(pca_model.explained_variance_ratio_.cumsum())):
                if pca_model.explained_variance_ratio_.cumsum()[n] > .8:
                    n_components = n
            pca_new = PCA(n)
            self.data = pca_new.fit_transform(self.data[self.data.columns])
            self.data = pd.DataFrame(self.data)

        print('numerical: ', self.numericals)
        print('categorical: ', self.categoricals)
        print('drop: ', self.drops)

    def cluster(self, kind='hdbscan', kmin=2, kmax=15):
        if kind == 'hdbscan':
            clusterer = hdbscan.HDBSCAN(metric='euclidean',
                                        allow_single_cluster=False,
                                        alpha=.5,
                                        min_cluster_size=15,  # int(len(self.data)*.005)&int(np.log(len(self.data))*15),
                                        min_samples=int(np.log(len(self.data))),
                                        prediction_data=True,
                                        cluster_selection_method='leaf')
            clusterer.fit(self.data)
            self.clusters['hdbscan'] = [np.argmax(x) for x in hdbscan.all_points_membership_vectors(clusterer)] #clusterer.labels_

        if kind == 'kmeans':
            sil_scores = [] # medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
            labels_list = []
            for k in range(kmin, kmax+1):
                kmeans = KMeans(n_clusters=k).fit(self.data)
                labels = kmeans.labels_
                sil_scores.append(silhouette_score(self.data, labels, metric='euclidean'))
                labels_list.append(labels)
                # sil_scores indexes: 0 is k=2, 1 is k=3, etc.
            ideal_k = sil_scores.index(max(sil_scores))
            # print(sil_scores)
            print('k=', ideal_k+kmin)
            self.clusters['kmeans'] = labels_list[ideal_k]
            del labels_list
            del sil_scores

        if kind == 'agglomerative':
            sil_scores = []
            labels_list = []
            for k in range(kmin, kmax+1):
                agglo = Agglo(n_clusters=k).fit(self.data)
                labels = agglo.labels_
                sil_scores.append(silhouette_score(self.data, labels, metric = 'euclidean'))
                labels_list.append(labels)
                # sil_scores indexes: 0 is k=2, 1 is k=3, etc.
            ideal_k = sil_scores.index(max(sil_scores))
            # print(sil_scores)
            print('k=', ideal_k+kmin)
            self.clusters['agglomerative'] = labels_list[ideal_k]
            del labels_list
            del sil_scores

    def tsne(self, kind='hdbscan', initialize=True):
        if initialize:
            self.projection = TSNE().fit_transform(self.data)
        color_palette = sns.color_palette('bright', np.array(self.clusters[kind]).max()+1)
        cluster_colors = [color_palette[x] if x >= 0
                          else (0.8, 0.8, 0.8)
                          for x in self.clusters[kind]]
        plt.scatter(*self.projection.T, s=50, linewidth=0, c=cluster_colors, alpha=0.25)

    def initial_dist(self):
        sns.set(style="whitegrid")
        for category in self.categoricals:
            g = sns.FacetGrid(self.data_original_categories, col=None)
            ax = sns.countplot(x=category,data=self.data_original_categories)
      
        for numerical in self.numericals:
            g = sns.FacetGrid(self.data_original_categories, col=None)
            ax = sns.distplot(self.data_original_categories[numerical])

    def clustered_dist(self, kind='hdbscan'):
        sns.set(style="whitegrid")
        for category in self.categoricals:
            g = sns.FacetGrid(self.data_original_categories, col=None)
            ax = sns.countplot(x=category,data=self.data_original_categories, hue=self.clusters[kind])
      
        for numerical in self.numericals:
            g = sns.FacetGrid(self.data_original_categories, col=None)
        for n in range(max(self.clusters[kind])+1):
            sns.distplot(self.data_original_categories[self.clusters[kind]==n][numerical])

    def metrics(self):
        for kind in self.clusters:
            print(kind + " David-Bouldin: " + str(davies_bouldin_score(self.data, self.clusters[kind])) + " (Lower is better, 0 min)")
            print(kind  + " Calinski-Harabasz: " + str(calinski_harabasz_score(self.data, self.clusters[kind])) + " (Higher is better)")
            print(kind + " Silhouette: " + str(silhouette_score(self.data, self.clusters[kind])) + " (Higher is better)")
            print(kind + " Alex Metric " + str(cluster_evaluation(self.clusters[kind])) + "(Higher is worse, 0 - 1)")


fara = pd.read_csv(my_drive + 'features.csv')
feature_dict = pd.read_csv(my_drive + 'faraday_dictionary.csv')

not_in_our_data = ['do_not_call', 'life_other_own_smart_phone_all', 
                   'house_number', 'person_first_name', 'apn', 
                   'person_last_name', 'glob', 'street', 'seller_name', 
                   'address_quality_code', 'house_number_and_street2', 
                   'person_first_name2', 'city2', 'postcode_zip42', 
                   'delivery_point_barcode', 'state2', 
                   'house_number_and_street', 'geochunk_zip2010_500000', 
                   'person_last_name2', 'postcode2', 'unit_number', 'person', 
                   'phone', 'the_geom_webmercator']
for each in not_in_our_data:
    feature_dict = feature_dict[feature_dict.columns][feature_dict['Code']!=each]

new_dataset = Dataset(fara)
numericals = list(feature_dict['Code'][feature_dict['Type']=='numeric'])
booleans = list(feature_dict['Code'][feature_dict['Type']=='boolean'])
categoricals = list(feature_dict['Code'][feature_dict['Type']=='categorical'])
dates = list(feature_dict['Code'][feature_dict['Type']=='date'])
locations = list(feature_dict['Code'][feature_dict['Type']=='location'])
drops = dates + locations + ['id']

fara[booleans] = fara[booleans].fillna(False)

new_dataset.assign_categories(categoricals, numericals, drops)

new_dataset.clean(find_categories=False, center=True, standardize=True, pca=False, iter_impute=True, drop_thresh=0)

"""yes = pd.read_csv(my_drive + 'features.csv')
new_dataset = Dataset(yes)
new_dataset.clean()
clustering_kind = 'hdbscan'
new_dataset.cluster(kind=clustering_kind, kmin=2, kmax=10)
#new_dataset.cluster(kind='kmeans')
new_dataset.tsne(kind=clustering_kind, initialize=True)
#new_dataset.initial_dist()
#new_dataset.metrics()
"""

# new_dataset.metrics()

