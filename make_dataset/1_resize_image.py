"""
このプログラムは入力された画像のサイズを298x298にリサイズします。
resize_image_folder_path にリサイズしたい画像をいれたフォルダのパスを入れてください。
save_resize_image_folder_path リサイズした画像を保存するフォルダのパスを入力してください。そのフォルダに298x298にリサイズした画像がpng形式で保存されます。
保存するときのファイル名は 298x298_image_何かの数字.png となります。
"""



from google.colab import drive
drive.mount('/content/drive')



import glob
import cv2

# このプログラムは2048x2048の画像を298x298にリサイズします
resize_image_folder_path = "/content/drive/Shareddrives/第5回中高生情報学研究コンテスト/手書き数字_2048x2048"
save_resize_image_folder_path = "/content/drive/Shareddrives/第5回中高生情報学研究コンテスト/手書き数字_298x298"
resize_image_file_paths = glob.glob(resize_image_folder_path + '/*.png')

for image_num, image_path in enumerate(resize_image_file_paths):
  cut_image = cv2.imread(image_path)
  cut_image = cv2.resize(cut_image, (298, 298))
  cv2.imwrite(
      save_resize_image_folder_path + f"/298x298_image_{image_num}.png", cut_image)