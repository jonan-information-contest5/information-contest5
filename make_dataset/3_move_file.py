"""
このプログラムはフォルダの中の画像の１割を別のフォルダに移動させます。
research_quantity_folder_path に移動させたい画像が入ったフォルダパスを入力してください。
input_folder_path に research_quantity_folder_path と同じパスを入れてください。
tomove_folder_path に移動先のフォルダのパスを入力してください。
"""

from google.colab import drive
drive.mount('/content/drive')


import shutil
import os

research_quantity_folder_path = "/content/drive/Shareddrives/第5回中高生情報学研究コンテスト/手書き数字のデータセット/学習用_精度判定用_両方"
folder = ["dataset.0", "dataset.1", "dataset.2", "dataset.3", "dataset.4", 
          "dataset.5", "dataset.6", "dataset.7", "dataset.8", "dataset.9"]

tmp_file_path = research_quantity_folder_path + "/" + folder[0]
min_quantity_file = sum(os.path.isfile(os.path.join(tmp_file_path, name)) for name in os.listdir(tmp_file_path))
for folder_name in folder:
  tmp_file_path = research_quantity_folder_path + "/" + folder_name
  file_quantity = sum(os.path.isfile(os.path.join(tmp_file_path, name)) for name in os.listdir(tmp_file_path))
  min_quantity_file = min(min_quantity_file, file_quantity)

print(min_quantity_file) # 一番画像が入ってないフォルダにある画像の数



import glob
import math
import random as rd

input_folder_path = "/content/drive/Shareddrives/第5回中高生情報学研究コンテスト/手書き数字のデータセット/学習用_精度判定用_両方"
tomove_folder_path = "/content/drive/Shareddrives/第5回中高生情報学研究コンテスト/手書き数字のデータセット/精度判定用"
move_count = math.ceil(min_quantity_file / 10)

for index1, folder_name in enumerate(folder):
  # ここからファイルを動かす
  file_paths = glob.glob(input_folder_path + '/' + folder_name + "/*.jpg")
  rd.shuffle(file_paths)
  for count in range(move_count):
    shutil.move(file_paths[count], tomove_folder_path + '/' + folder_name)