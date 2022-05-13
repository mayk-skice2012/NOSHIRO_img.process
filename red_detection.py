from matplotlib import image
import numpy as np
import cv2
from time import sleep
from PIL import Image
import matplotlib.pyplot as plt


#実際に用いる関数(sampleimage:取得した画像）
def main(sampleimage):
    
    Rotation = rot_get(center_get(sampleimage),size_get(sampleimage),4.8,4.5)
    Occupancy = occ_get(sampleimage)
    
    if Rotation != 'cannot find':
        return [Rotation, Occupancy]
    
    else:
        return [0,0]



# テスト関数(sampleimage:取得した画像, f_mm:焦点距離[mm], diagonal_mm:対角線[mm])
def test(sampleimage, f_mm, diagonal_mm):

    image_output(red_masks_get(sampleimage),'sample_red.jpg')
    image_output(gray_get(sampleimage),'sample_gray.jpg')
    image_output(binary_get(sampleimage),'sample_binary.jpg')
    center_output(sampleimage, 'sample_center.jpg', center_get(sampleimage))

    print('Size:', size_get(sampleimage))
    print('Occupancy:', occ_get(sampleimage)*100, '%')
    print('Center:',center_get(sampleimage))
    print('Rotation:',rot_get(center_get(sampleimage),size_get(sampleimage),f_mm,diagonal_mm))

# ESP32-CAM: f_mm = 4.8, diagonal_mm = 4.5
# iPhone11: f_mm = 13, diagonal_mm = 7.14



# 画像を出力する関数
def image_output(image_data,image_name):
    img_arrange = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)  # グラフにするために形式変換
    plt.imshow(img_arrange)  
    plt.savefig(image_name)  # 画像を出力する



# 重心座標と抽出画像を出力する関数
def center_output(sampleimage, image_name, center):
    img_arrange = cv2.cvtColor(red_masks_get(sampleimage), cv2.COLOR_BGR2RGB)  # グラフにするために形式変換
    plt.imshow(img_arrange)  # 画像をグラフに貼り付け

    x,y = center[0],center[1]
    plt.plot(x,y,marker='.')  # 重心座標をグラフに張り付け  

    plt.savefig(image_name)  # グラフを表示



# 画像を表示する関数
def image_plot(sampleimage):
    image = cv2.imread(sampleimage)  # 画像を読み込み
    img_arrange = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # グラフにするために形式変換
    plt.imshow(img_arrange)  # 画像をグラフに貼り付け
    plt.show()  # グラフを表示



# HSVで特定の色を抽出する関数
def hsvExtraction(image, hsvLower, hsvUpper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 画像をHSVに変換
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)  # HSVからマスクを作成
    result = cv2.bitwise_and(image, image, mask=hsv_mask)  # 元画像とマスクを合成
    return result



# 抽出画像データを取得する関数
def red_masks_get(sampleimage):

    # HSVでの色抽出
    hsvLower_0 = np.array([0, 60, 50])    # 抽出する色の下限0(赤の抽出のため二つにわけて合成が必要)
    hsvLower_1 = np.array([170, 60, 50])  # 抽出する色の下限1(赤の抽出のため二つにわけて合成が必要)
    hsvUpper_0 = np.array([10, 255, 255])    # 抽出する色の上限0(赤の抽出のため二つにわけて合成が必要)
    hsvUpper_1 = np.array([179, 255, 255])   # 抽出する色の上限1(赤の抽出のため二つにわけて合成が必要)

    hsvResult_0 = hsvExtraction(sampleimage, hsvLower_0, hsvUpper_0)  # 画像0を作成
    hsvResult_1 = hsvExtraction(sampleimage, hsvLower_1, hsvUpper_1)  # 画像1を作成

    hsvResult = hsvResult_0 | hsvResult_1  # 画像を統合
    
    return hsvResult



# グレースケール画像データを取得する関数
def gray_get(sampleimage):
    img = red_masks_get(sampleimage)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # グレースケールで画像を読み込み
    return img_gray



# 二値画像データを取得する関数
def binary_get(sampleimage):
    ret, img_binary = cv2.threshold(gray_get(sampleimage), 10, 255, cv2.THRESH_BINARY)  # 2値画像に変換
    return img_binary



# 占有率を求める関数
def occ_get(sampleimage):
    pixel_number = np.size(binary_get(sampleimage))  # 全ピクセル数
    pixel_sum = np.sum(binary_get(sampleimage))  # 輝度の合計数
    white_pixel_number = pixel_sum/255  # 白のピクセルの数

    return white_pixel_number / pixel_number  # 占有率を計算



# 重心座標を求める関数
def center_get(sampleimage):
    mu = cv2.moments(binary_get(sampleimage), False)  # 重心関数作成
    
    if mu["m00"] != 0:
        x,y= int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])  # 重心座標の作成
        return [x,y]  # 重心座標を返り値に設定
    
    else:
        return 'cannot find'





# 画像のサイズの取得
def size_get(sampleimage):
    height, width, channel = sampleimage.shape
    return [width,height]



# 回転角を取得(center:重心座標, size:画像サイズ, f_mm:焦点距離[mm], diagonal_mm:撮像素子の対角線[mm])
def rot_get(center, size, f_mm, diagonal_mm):  

    if center != 'cannot find':
        width_mm = diagonal_mm * size[0]/np.sqrt(size[0]*size[0] + size[1]*size[1])
        sita_rad = np.arctan((width_mm * (size[0] / 2 - center[0]) / size[0]) / f_mm)  # 回転角θ[rad]を導出
        sita = 180*sita_rad/np.pi

        return sita
    
    else:
        return 'cannot find'



##############################################################################################################

#1.テストをする場合

# 解析したい画像を入力
img = "red_cone_1.jpeg"



# 画像の種類によってmain関数を選択

test(img, 4.8, 4.5)  # ESP32-CAM用
#test(img, 4, 7.14)  # iPhone11用