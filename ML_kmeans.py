#interpreter: anaconda3

from matplotlib import image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from IPython.display import Image, display
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import collections


#画像読み込み
#sample 画像 = "red_cone_1.jpeg"
img = cv2.imread("red_cone_1.jpeg") # グラフにするために形式変換


#画像出力
#def image_output(image_data, image_name): 
plt.imshow(np.array(img))
#plt.savefig(image_name)  # 画像を出力する
plt.show()

#kmeans クラスタリングを使った色の分析
#def image_process(image_data):
colors = img.reshape(-1, 3).astype(np.float32) #3次元データ(RGB値のarray)に変換
n_clusters = 6

# 最大反復回数: 10、移動量の閾値: 1.0
criteria = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0
ret, labels, centers = cv2.kmeans(
    colors, n_clusters, None, criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS
)


print(f"ret: {ret:.2f}, label: {labels.shape}, center: {centers.shape}")

labels = labels.squeeze(axis=1)  # (N, 1) -> (N,)
centers = centers.astype(np.uint8)  # float32 -> uint8

# 各クラスタに属するサンプル数を計算する。
_, counts = np.unique(labels, axis=0, return_counts=True)

# 可視化する。
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 3))
fig.subplots_adjust(wspace=0.5)

# matplotlib の引数の仕様上、[0, 1] にして、(R, G, B) の順番にする。
bar_color = centers[:, ::-1] / 255
bar_text = list(map(str, centers))

# 画像を表示する。
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_axis_off()

# ヒストグラムを表示する。
ax2.barh(np.arange(n_clusters), counts, color=bar_color, tick_label=bar_text)
plt.show()

#赤色を抽出する
print(counts[5]) #red_cone_1.jpegだと赤は16250

#赤がどのくらいだと