"""
このプログラムはサイズが298x298の画像を100枚の画像に分割し、同じ数字のみを同じフォルダに入れるプログラムです。
cut_image_folder_path に切り分けたい画像が入ったフォルダのパスを入力してください。
save_cut_image_folder_path に切り分けた画像を入れたいフォルダのパスを入力してください（この際、そのフォルダの中に dataset.0 ~ dataset.9のフォルダを全部で１０個あらかじめ作っておいてください）。
保存される画像のファイル名は なんかの数字.png です。
"""

from google.colab import drive
drive.mount('/content/drive')


# 298x298 の画像の分割のみ使えます

import cv2
import matplotlib.pyplot as plt
import glob

cut_image_folder_path = "/content/drive/Shareddrives/第5回中高生情報学研究コンテスト/手書き数字_298x298"
save_cut_image_folder_path = "/content/drive/Shareddrives/第5回中高生情報学研究コンテスト/手書き数字のデータセット/学習用_精度判定用_両方"

cut_image_file_paths = glob.glob(cut_image_folder_path + "/*.png")

for index, image_path in enumerate(cut_image_file_paths):
  image_handle = cv2.imread(image_path,-1) # 一つずつ読み込む
  for h in range(10):
    for w in range(10):
      cut_image = image_handle[30*h : 30*h+28, 30*w : 30*w+28] #img[top : bottom, left : right]
      cv2.imwrite(save_cut_image_folder_path + f"/dataset.{h}/{10*index+w}.jpg", cut_image) # {10*index+w}.jpg として保存