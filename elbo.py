#クラスター数を決める
#サンプル画像とさほど変わらないはず

from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import cv2

img = cv2.cvtColor(cv2.imread("red_cone_1.jpeg"), cv2.COLOR_BGR2RGB) # グラフにするために形式変換

X = img.data[1, 1]
y = img.target

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show()
