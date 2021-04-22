# 狙い．
![image](https://user-images.githubusercontent.com/43159778/115698015-1f963f80-a39f-11eb-80f3-a1a562169b50.png)
dataset/train/image/2.jpgの画像  
この画像にはepitheliumとneutrophils二つが含まれている．この画像に対するマスク画像は  
epitheliumのマスク画像，  
![image](https://user-images.githubusercontent.com/43159778/115698064-2d4bc500-a39f-11eb-8ef4-24d6675edfe3.png)  

neutrophilsのマスク画像  
![image](https://user-images.githubusercontent.com/43159778/115698084-3341a600-a39f-11eb-816c-0b1b24a46c87.png)

それぞれのマスク画像をdataset.pyで指定したクラスidと紐づけながらロードしたい．
期待している出力．  
![image](https://user-images.githubusercontent.com/43159778/115698695-deeaf600-a39f-11eb-8b10-14731c925acf.png)


## dataset.py
めんだこさんのDatasetクラスをモジュール化したもの．追加したコードは  
 self.add_class('cell_dataset', 1, 'cell')    
から  
        self.add_class('cell_dataset', 1, 'neutrophils')  
        self.add_class('cell_dataset', 2, 'epithelium')  
のみ．

## load_jpg.ipynb
上記のdataset.pyをインポートして，画像を読み込み，出力して確認する用のコード．  
Args ; 
      mode : datasetフォルダ直下のtrainフォルダから読み取るか，validフォルダから読み取るか．最終的にはtrainフォルダのような構造からimageとmaskを取りたい．

