# ドライブのマウント
from google.colab import drive
drive.mount('/content/drive')


# 使用するライブラリのインポート
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import time
import pickle




# 精度判定の準備
check_data_path = "/content/drive/Shareddrives/第5回中高生情報学研究コンテスト/手書き数字のデータセット/精度判定用画像"
folder = ["dataset.0", "dataset.1", "dataset.2", "dataset.3", "dataset.4", 
          "dataset.5", "dataset.6", "dataset.7", "dataset.8", "dataset.9"]
my_check_images = np.empty(((1, 8, 8)), dtype=int) #学習用画像データ
my_check_labels = [] # 学習用ラベルデータ

for index, folder_name in enumerate(folder):
  start_time = time.time()
  read_data = check_data_path + '/' + folder_name
  jpg_files = glob.glob(read_data + "/*.jpg")
  for jpg_file in jpg_files:

    img = Image.open(jpg_file)
    img = img.convert('L') # グレースケールに変換
    img = img.resize((8, 8))
    img = np.array(img)
    img = np.reshape(img, (1, 8, 8)) # ３次元配列に変換
    my_check_images = np.append(my_check_images, img, axis=0)
    my_check_labels = np.append(my_check_labels, index)

  print(time.time() - start_time)
  print(folder_name, "complete")

my_check_images = np.delete(my_check_images, 0, 0)



# 学習用画像による判定

sav_file_name = "predicted_by_学習用画像_8x8.sav"
sav_file_path = f"/content/drive/Shareddrives/第5回中高生情報学研究コンテスト/データセット作成用/sav_files/{sav_file_name}"
clf = pickle.load(open(sav_file_path,'rb'))
my_check_images = my_check_images
my_check_labels = my_check_labels

correct_ans_list  = [0] * 10 # 正解の数を数える
incorrect_ans_list = [0] * 10 # 不正解の数を数える
# 上のは[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] の順に入る

for i in range(len(my_check_labels)):
  check_image = my_check_images[i]
  check_label = my_check_labels[i]
  check_image = np.array(check_image, dtype=np.uint8)
  check_image = check_image.reshape(-1, 64)
  res = clf.predict(check_image)

  if int(res) == int(check_label):
    correct_ans_list[int(check_label)] += 1
  else:
    incorrect_ans_list[int(check_label)] += 1

print(sav_file_name, "による判定結果")
print("手書き数字のデータセット/精度判定用画像 の画像を 8x8に圧縮したものを判定")
print("判定する画像は256階調")
for i in range(10):
  print(f"{i}は　正解{correct_ans_list[i]}こ　不正解{incorrect_ans_list[i]}こ")
  print(f"正解率は{correct_ans_list[i] / (correct_ans_list[i] + incorrect_ans_list[i])}")
print("です")




# 水増しした画像による判定

sav_file_name = "predicted_by_水増しした画像_8x8.sav"
sav_file_path = f"/content/drive/Shareddrives/第5回中高生情報学研究コンテスト/データセット作成用/sav_files/{sav_file_name}"
clf = pickle.load(open(sav_file_path,'rb'))
my_check_images = my_check_images
my_check_labels = my_check_labels

correct_ans_list  = [0] * 10 # 正解の数を数える
incorrect_ans_list = [0] * 10 # 不正解の数を数える
# 上のは[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] の順に入る

for i in range(len(my_check_labels)):
  check_image = my_check_images[i]
  check_label = my_check_labels[i]
  check_image = np.array(check_image, dtype=np.uint8)
  check_image = check_image.reshape(-1, 64)
  res = clf.predict(check_image)

  if int(res) == int(check_label):
    correct_ans_list[int(check_label)] += 1
  else:
    incorrect_ans_list[int(check_label)] += 1

print(sav_file_name, "による判定結果")
print("手書き数字のデータセット/精度判定用画像 の画像を 8x8に圧縮したものを判定")
print("判定する画像は256階調")
for i in range(10):
  print(f"{i}は　正解{correct_ans_list[i]}こ　不正解{incorrect_ans_list[i]}こ")
  print(f"正解率は{correct_ans_list[i] / (correct_ans_list[i] + incorrect_ans_list[i])}")
print("です")




# sklearnによる判定

sav_file_name = "predicted_by_sklearn_8x8.sav"
sav_file_path = f"/content/drive/Shareddrives/第5回中高生情報学研究コンテスト/データセット作成用/sav_files/{sav_file_name}"
clf = pickle.load(open(sav_file_path,'rb'))
my_check_images = my_check_images
my_check_labels = my_check_labels

correct_ans_list  = [0] * 10 # 正解の数を数える
incorrect_ans_list = [0] * 10 # 不正解の数を数える
# 上のは[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] の順に入る

for i in range(len(my_check_labels)):
  check_image = my_check_images[i]
  check_label = my_check_labels[i]
  check_image = np.array(check_image, dtype=np.uint8)
  check_image = 15 - check_image // 16
  check_image = check_image.reshape(-1, 64)
  res = clf.predict(check_image)

  if int(res) == int(check_label):
    correct_ans_list[int(check_label)] += 1
  else:
    incorrect_ans_list[int(check_label)] += 1

print(sav_file_name, "による判定結果")
print("手書き数字のデータセット/精度判定用画像 の画像を 8x8に圧縮したものを判定")
print("判定する画像は16階調")
for i in range(10):
  print(f"{i}は　正解{correct_ans_list[i]}こ　不正解{incorrect_ans_list[i]}こ")
  print(f"正解率は{correct_ans_list[i] / (correct_ans_list[i] + incorrect_ans_list[i])}")
print("です")




# 学習用画像による判定

sav_file_name = "predicted_by_学習用画像_8x8.sav"
sav_file_path = f"/content/drive/Shareddrives/第5回中高生情報学研究コンテスト/データセット作成用/sav_files/{sav_file_name}"
clf = pickle.load(open(sav_file_path,'rb'))
my_check_images = my_check_images
my_check_labels = my_check_labels

correct_ans_list  = [0] * 10 # 正解の数を数える
incorrect_ans_list = [0] * 10 # 不正解の数を数える
# 上のは[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] の順に入る

for i in range(len(my_check_labels)):
  check_image = my_check_images[i]
  check_image = check_image // 16
  check_image = check_image * 16
  check_label = my_check_labels[i]
  check_image = np.array(check_image, dtype=np.uint8)
  check_image = check_image.reshape(-1, 64)
  res = clf.predict(check_image)

  if int(res) == int(check_label):
    correct_ans_list[int(check_label)] += 1
  else:
    incorrect_ans_list[int(check_label)] += 1

print(sav_file_name, "による判定結果")
print("手書き数字のデータセット/精度判定用画像 の画像を 8x8に圧縮したものを判定")
print("判定する画像は16階調に変換")
for i in range(10):
  print(f"{i}は　正解{correct_ans_list[i]}こ　不正解{incorrect_ans_list[i]}こ")
  print(f"正解率は{correct_ans_list[i] / (correct_ans_list[i] + incorrect_ans_list[i])}")
print("です")




# 水増しした画像による判定

sav_file_name = "predicted_by_水増しした画像_8x8.sav"
sav_file_path = f"/content/drive/Shareddrives/第5回中高生情報学研究コンテスト/データセット作成用/sav_files/{sav_file_name}"
clf = pickle.load(open(sav_file_path,'rb'))
my_check_images = my_check_images
my_check_labels = my_check_labels

correct_ans_list  = [0] * 10 # 正解の数を数える
incorrect_ans_list = [0] * 10 # 不正解の数を数える
# 上のは[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] の順に入る

for i in range(len(my_check_labels)):
  check_image = my_check_images[i]
  check_image = check_image // 16
  check_image = check_image * 16
  check_label = my_check_labels[i]
  check_image = np.array(check_image, dtype=np.uint8)
  check_image = check_image.reshape(-1, 64)
  res = clf.predict(check_image)

  if int(res) == int(check_label):
    correct_ans_list[int(check_label)] += 1
  else:
    incorrect_ans_list[int(check_label)] += 1

print(sav_file_name, "による判定結果")
print("手書き数字のデータセット/精度判定用画像 の画像を 8x8に圧縮したものを判定")
print("判定する画像は16階調に変換")
for i in range(10):
  print(f"{i}は　正解{correct_ans_list[i]}こ　不正解{incorrect_ans_list[i]}こ")
  print(f"正解率は{correct_ans_list[i] / (correct_ans_list[i] + incorrect_ans_list[i])}")
print("です")