# ドライブのマウント
from google.colab import drive
drive.mount('/content/drive')


# 使用するライブラリのインポート
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob


train_data_path = "/content/drive/Shareddrives/第5回中高生情報学研究コンテスト/手書き数字のデータセット/学習用画像"
folder = ["dataset.0", "dataset.1", "dataset.2", "dataset.3", "dataset.4", 
          "dataset.5", "dataset.6", "dataset.7", "dataset.8", "dataset.9"]
my_train_images = np.empty(((1, 28, 28)), dtype=int) #学習用画像データ
my_train_labels = [] # 学習用ラベルデータ

for index, folder_name in enumerate(folder):
  read_data = train_data_path + '/' + folder_name
  jpg_files = glob.glob(read_data + "/*.jpg")
  for jpg_file in jpg_files:

    img = Image.open(jpg_file)
    img = img.convert('L') # グレースケールに変換
    img = np.array(img)
    img = np.reshape(img, (1, 28, 28)) # ３次元配列に変換
    my_train_images = np.append(my_train_images, img, axis=0)
    my_train_labels = np.append(my_train_labels, index)

my_train_images = np.delete(my_train_images, 0, 0)


print(len(my_train_labels))




from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

x = my_train_images
y = my_train_labels
# 2次元配列の画像データを1次元配列に変換
x = x.reshape((-1,784))
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)
clf = SVC()
# 学習
clf.fit(x_train, y_train)
# 予測
y_pred = clf.predict(x_test)

# 正解率の確認
print("\n正解率＝", accuracy_score(y_test, y_pred))



# 学習済みモデルの保存
import pickle

save_folder_path = "/content/drive/Shareddrives/第5回中高生情報学研究コンテスト/データセット作成用/sav_files"
file_name = "predicted_by_学習用画像.sav"

pickle.dump(clf, open(save_folder_path + '/' + file_name,'wb'))  



# 精度判定の準備
check_data_path = "/content/drive/Shareddrives/第5回中高生情報学研究コンテスト/手書き数字のデータセット/精度判定用画像"
folder = ["dataset.0", "dataset.1", "dataset.2", "dataset.3", "dataset.4", 
          "dataset.5", "dataset.6", "dataset.7", "dataset.8", "dataset.9"]
my_check_images = np.empty(((1, 28, 28)), dtype=int) #学習用画像データ
my_check_labels = [] # 学習用ラベルデータ

for index, folder_name in enumerate(folder):
  read_data = check_data_path + '/' + folder_name
  jpg_files = glob.glob(read_data + "/*.jpg")
  for jpg_file in jpg_files:

    img = Image.open(jpg_file)
    img = img.convert('L') # グレースケールに変換
    img = np.array(img)
    img = np.reshape(img, (1, 28, 28)) # ３次元配列に変換
    my_check_images = np.append(my_check_images, img, axis=0)
    my_check_labels = np.append(my_check_labels, index)

my_check_images = np.delete(my_check_images, 0, 0)



print(len(my_check_labels))


# 精度判定用

sav_file_path = "/content/drive/Shareddrives/第5回中高生情報学研究コンテスト/データセット作成用/sav_files/predicted_by_学習用画像.sav"
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
  check_image = check_image.reshape(-1, 784)
  res = clf.predict(check_image)

  if int(res) == int(check_label):
    correct_ans_list[int(check_label)] += 1
  else:
    incorrect_ans_list[int(check_label)] += 1

for i in range(10):
  print(f"{i}は　正解{correct_ans_list[i]}こ　不正解{incorrect_ans_list[i]}こ")
  print(f"正解率は{correct_ans_list[i] / (correct_ans_list[i] + incorrect_ans_list[i])}")
print("です")