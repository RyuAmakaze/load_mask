import pathlib
import cv2
from PIL import Image
from mrcnn import utils
from mrcnn.model import log
import glob
import os
import copy
import numpy as np

global masks

class ShapesDataset(utils.Dataset):

    def load_dataset(self, dataset_dir):
        """ データセットを登録
        """
        #: データセット名、クラスID、クラス名
        self.add_class('cell_dataset', 1, 'neutrophils')
        self.add_class('cell_dataset', 2, 'epithelium')

        #ファイルを取得．
        images = glob.glob(os.path.join(dataset_dir, "image", "*.png"))
        masks = glob.glob(os.path.join(dataset_dir, "mask", "*.png"))

        for image_path, mask_path in zip(images, masks):
            print(mask_path)
            image_path = pathlib.Path(image_path)
            mask_path = pathlib.Path(mask_path)
            assert image_path.name == mask_path.name, 'データセット名不一致'

            image = Image.open(image_path)
            height = image.size[0]
            width = image.size[1]

            mask = Image.open(mask_path)
            assert image.size == mask.size, "サイズ不一致"

            self.add_image(
                'cell_dataset',
                path=image_path,
                image_id=image_path.stem,
                mask_path=mask_path,
                width=width, height=height)



    def load_mask(self, image_id):
        """マスクデータとクラスidを生成する
        """
        print("hoge")
        image_info = self.image_info[image_id]
        print(image_info)
        if image_info["source"] != 'cell_dataset':
            return super(self.__class__, self).load_mask(image_id)

        mask_path = image_info['mask_path']
        mask, cls_idxs = blob_detection(str(mask_path))

        return mask, cls_idxs

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == 'cell_dataset':
            return info
        else:
            super(self.__class__, self).image_reference(image_id)


def blob_detection(mask_path):
    mask = cv2.imread(mask_path, 0)
   #: 念のためもう一度二値化
    _, mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)

    label = cv2.connectedComponentsWithStats(mask)
    data = copy.deepcopy(label[1])

    labels = []
    for label in np.unique(data):
        #: ラベル0は背景
        if label == 0:
            continue
        else:
            labels.append(label)

    mask = np.zeros((mask.shape)+(len(labels),), dtype=np.uint8)
    #print("hoge2")
    #print(mask)
    for n, label in enumerate(labels):
        mask[:, :, n] = np.uint8(data == label)

    cls_idxs = np.ones([mask.shape[-1]], dtype=np.int32)
    #print(cls_idxs)

    return mask, cls_idxs
