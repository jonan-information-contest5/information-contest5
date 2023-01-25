#move_centerの引数には画像のファイルパスを入れて下さい
#中央寄せされた画像を返します


import numpy as np
from PIL import Image


def get_margin(image):
  margin=[0,0,0,0] #top bottom right left

  for f,g in enumerate((image, np.rot90(image))):  #上下右左の順で余白を取得 
    for h,i in enumerate((g,g[::-1])):              
      for j,k in enumerate(i):
        if np.any(k==0)==True:      #黒色(文字色)があれば余白を確定
          margin[f*2+h]=j
          break
  return margin
  

def move_center(image):
  image = Image.open(image)
  image = np.array(image)   #ndarrayに変換
  margin = get_margin(image)  #上下左右の余白の取得
  am_move=[(margin[1]-margin[0])//2,(margin[2]-margin[3])//2]  #移動量の計算[縦 , 横]
  image=np.roll(image,(am_move[0],am_move[1]),axis=(0,1)) #移動
  return image

