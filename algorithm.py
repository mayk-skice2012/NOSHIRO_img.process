from gettext import install
from re import S
from matplotlib import image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from IPython.display import Image, display
from pytz import NonExistentTimeError

import pyserial
from geopy.distance import geodesic #

ser = pyserial.Serial()
ser.port = "****"     
ser.baudrate = 1000
ser.timeout = None
ser.setDTR(False)   
ser.open()  

while True:
    line = ser.readline()
    print(line)
ser.close()

#GPSセンサから緯度と経度を取得
position_cone = (0,0)
position_i = (a,b) #初期位置
dis = geodesic(position_i, position_cone).m
print(dis)



#回転
#機体の中心をz軸とした時の回転 or 機体の横をx軸とした回転
#機体の中心をz軸とした時の回転とする





#csvファイルに出力

import csv
"""
    csvに測位結果を書き込む
    :param path: 書き込み先
    :type path: str
    :param rec_time: 測位時刻 YYYYMMDD HH:mm:ss
    :type rec_time: str
    :param lon: 経度 世界測地系 度数
    :type lon: float
    :param lat: 緯度 世界測地系 度数
    :type lat: float
    :param alt: 海抜 メートル
    :type alt: float
    :param speed: スピード
    :type speed: float
    :param satellites_used: 測位衛星番号
    :type satellites_used: list
    """

