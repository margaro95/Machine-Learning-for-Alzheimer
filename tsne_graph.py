"""Script to visualize t-SNE from our data."""
import numpy as np
import matplotlib.pyplot as plt

from sklearn import manifold, ensemble, decomposition
from createTargets import createTargets_DCLvsControl
from createDataset import addaptDataset
from sklearn.decomposition import PCA
from matplotlib import offsetbox
import pdb

dataset = np.load("dataset_alzheimer.npy")
X = addaptDataset(dataset, np.array([9*30, 9*30, 9*29]), 4005, 16, 9)
X = X - np.mean(X, axis=0)
pca = PCA()
X = pca.fit(X).transform(X)
targets = createTargets_DCLvsControl(np.array([30*9, 30*9, 29*9]), 16, 9)


def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(int(targets[i]))+"-"+str(i),
                 color=plt.cm.Set1(targets[i] / 8),
                 fontdict={'weight': 'bold', 'size': 9})

    # if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        # shown_images = np.array([[1., 1.]])  # just something big
        # for i in range(X.shape[0]):
            # pdb.set_trace()
            # dist = np.sum((X[i] - shown_images) ** 2, 1)
            # if np.min(dist) < 4e-3:
                # don't show points that are too close
                # continue
            # shown_images = np.r_[shown_images, [X[i]]]
            # imagebox = offsetbox.AnnotationBbox(
                # offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                # X[i])
            # ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


# tsne = manifold.TSNE(n_components=2, random_state=0)
# X_tsne = tsne.fit_transform(X)
# plot_embedding(X_tsne)

hasher = ensemble.RandomTreesEmbedding(n_estimators=400,
                                       max_depth=None)
X_transformed = hasher.fit_transform(X)
pca = decomposition.TruncatedSVD(n_components=2)
X_reduced = pca.fit_transform(X_transformed)
plot_embedding(X_reduced)

plt.show()
