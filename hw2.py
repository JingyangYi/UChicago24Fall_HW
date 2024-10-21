

import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

headers = ['x' + ('00' + str(i))[-3:] for i in range(785)]  # 785 data columns
# 'x000 (first column is y target label, value 0-9), x001, x002, etc there are 28x28=784 pixels columns,
filepath = f'../data/mnist_train.csv'

df = pd.read_csv(filepath, names=headers)

# Check the shape and data type, make sure everything looks fine.
print(df.shape, df.dtypes)
df.head(5)

d_first = df.iloc[0,1:].values.reshape(28,28)
d_last = df.iloc[-1,1:].values.reshape(28,28)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(d_first)
ax1.set_title('First Digit')
ax2.imshow(d_last)
ax2.set_title('Last Digit')
plt.show()

pca = PCA(n_components=2)
def pca_draw(df, n_sample=500):
    df = df.iloc[:n_sample,:]
    mnist = df.iloc[:,1:].values.reshape(-1, 28 * 28)
    nor_mnist = (mnist - mnist.mean(axis=1, keepdims=True)) / mnist.std(axis=1, keepdims=True)
    labels = df.iloc[:,0].values.reshape(-1, 1)
    transformed1 = pca.fit_transform(mnist)
    transformed2 = pca.fit_transform(nor_mnist)
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10, 5))
    im1 = ax1.scatter(transformed1[:, 0], transformed1[:, 1], c=labels, cmap='viridis', marker='.')
    im2 = ax2.scatter(transformed2[:, 0], transformed2[:, 1], c=labels, cmap='viridis', marker='.')
    ax1.set_title(f"PCA on MNIST with normalization=False ")
    ax2.set_title(f"PCA on MNIST with normalization=True ")
    bar = fig.colorbar(im1, ax=[ax1 , ax2])
    bar.set_label('Intensity')
    plt.show()
pca_draw(df)
print("Discussion 1: we can see from the two graph that using simple n_components=2 PCA didn't have a good performance on "
      "both the normalized and non-normalized data. Some digits, like 0, 1, have relatively good clustering and can be "
      "recognized clearly on the graphs. Digits 3,4,5 are mixed together as a whole cluster, and digits 7,8,9 also mixed"
      "together as another cluster. These results indicate that simple 2-dimensional PCA is not a good method here, which"
      "only shown very limited clustering quality.")


# The silhouette score measures how similar a data point is to its own cluster (cohesion) compared to other clusters
# (separation). To speed up the training, we use PCA to reduce the dimension of the mnist pictures. Here we perform a
# grid search over the n_clusters
pca = PCA(n_components=100)
mnist = df.iloc[:,1:].values.reshape(-1, 28 * 28)
nor_mnist = (mnist - mnist.mean(axis=1, keepdims=True)) / mnist.std(axis=1, keepdims=True)
labels = df.iloc[:,0].values.reshape(-1, 1)
X_pca = pca.fit_transform(nor_mnist)
n_cluster_range = range(2, 20)
silhouette_scores = []
inertia_scores = []
for n in n_cluster_range:
    kmeans = KMeans(n_clusters=n, n_init=10, random_state=0).fit(X_pca)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X_pca, labels)
    print(f'The average silhouette_score for {n} clusters is {silhouette_avg}')
    silhouette_scores.append(silhouette_avg)
    inertia_scores.append(kmeans.inertia_)
print(f"Based on Silhouette score, {silhouette_scores.index(max(silhouette_scores))} clusters in the optimal")
print(f"Based on inertia score, {inertia_scores.index(max(inertia_scores))} clusters in the optimal")
print("----"*30)
print("Discussion: The results show that 11 is the best n_cluster, however, we can see that the corresponding ")