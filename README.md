# *これは｢第5回中高生情報学コンテスト｣に応募したときに用いたソースコードです。*

### 手書き数字判定の精度向上を目指したものです。



## 以下、各ファイルの概要



## make_datasetには高校生の手書き数字でデータセットを作った際のソースコードが入っています。

 #### 1_resize_image      
　　2048×2048の画像を298×298に変更しています

 #### 2_cut_image　　　    
　　1で変更した画像を各数字で分割し、各フォルダに転送しています

 #### 3_move_file         
　　精度判定用の画像として2の1割を別フォルダに転送しています

 #### 4_SVM_28*28         
　　サポートべクタマシンを用いて学習させモデルを作成、精度判定を行っています

 #### 7_make_sklearn_model  
　　sklearn内データを用いたモデルの作成を行っています

 #### 8_SVM_8*8           
　　4と同様ですが、8*8に圧縮して処理しています

 #### 10_compare_highschool_foreign　　
　　sklearn,高校生のモデルでそれぞれ比較しています


## move_centerには手書き数字の中央寄せ処理を行った際の自作関数が入っています。

## move_center_moduleはmove_centerをimport するために作ったフォルダです。
　import sys

  root_path = " "

  sys.path.append(root_path)

  import move_center as mc

 root_pathにmove_center_moduleのパスを指定し、上のコードを実行することでmc.move_center()を使うことができます

