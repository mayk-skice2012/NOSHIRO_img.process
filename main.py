from matplotlib import image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


#画像読み込み
#sample 画像 = "red_cone_1.jpeg"
img = cv2.cvtColor(cv2.imread("red_cone_1.jpeg"), cv2.COLOR_BGR2RGB) # グラフにするために形式変換

###################################################
def test(sampleimage, f_mm, diagonal_mm):

    image_output(red_get(sampleimage),'sample_red.jpg')
    image_output(gray_get(sampleimage),'sample_gray.jpg')
    image_output(binary_get(sampleimage),'sample_binary.jpg')

    print('Size:', size_get(sampleimage))
    print('Occupancy:', kmeans_occupy(sampleimage)*100, '%')
    print('depth:',depth_get(center_get(sampleimage),size_get(sampleimage),f_mm,diagonal_mm))


#画像出力
def image_output(image_data, image_name): 
    plt.imshow(np.array(image_data))
    plt.savefig(image_name)  # 画像を出力する
    plt.show()
    cv2.waitKey(1000) #画像を1秒表示後、消える
    cv2.destroyAllWindows()

# 画像のサイズの取得
def size_get(sampleimage):
    height, width, channel = sampleimage.shape
    return [width,height]


#占有率をkmeans から取得
def kmeans_occupy(sampleimage):
    return

#kmeans クラスタリングを使った色の分析
def image_process(image_data):
    flatten = image_data.reshape(-1,3) #3次元データに変換
    pred = KMeans(n_clusters=5).fit(flatten) #5色にクラスタリング

    out = zip(pred.labels_, flatten)
    clu0 = np.array([data.tolist() for label, data in zip(pred.labels_,flatten) if label==0])
    clu1 = np.array([data.tolist() for label, data in zip(pred.labels_,flatten) if label==1])
    clu2 = np.array([data.tolist() for label, data in zip(pred.labels_,flatten) if label==2])
    clu3 = np.array([data.tolist() for label, data in zip(pred.labels_,flatten) if label==3])
    clu4 = np.array([data.tolist() for label, data in zip(pred.labels_,flatten) if label==4])


    # センターカラーの取得
    color0 = pred.cluster_centers_[0] / 255
    color1 = pred.cluster_centers_[1] / 255
    color2 = pred.cluster_centers_[2] / 255
    color3 = pred.cluster_centers_[3] / 255
    color4 = pred.cluster_centers_[4] / 255


    # プロット
    fig = plt.figure()
    ax = Axes3D(fig)
    # 軸ラベルの設定
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    # 表示範囲の設定
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)
    ax.plot(clu0[:,0], clu0[:,1], clu0[:,2], "o", color=color0, ms=4, mew=0.5)
    ax.plot(clu1[:,0], clu1[:,1], clu1[:,2], "o", color=color1, ms=4, mew=0.5)
    ax.plot(clu2[:,0], clu2[:,1], clu2[:,2], "o", color=color2, ms=4, mew=0.5)
    ax.plot(clu3[:,0], clu3[:,1], clu3[:,2], "o", color=color3, ms=4, mew=0.5)
    ax.plot(clu4[:,0], clu4[:,1], clu4[:,2], "o", color=color4, ms=4, mew=0.5)

    plt.show()
    cv2.waitKey(1000) #画像を1秒表示後、消える
    cv2.destroyAllWindows()


#赤色検出
def red_get(sampleimage):
    return 

def depth_get(sampleimage):
    return
