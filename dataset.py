import copy
import glob
import os
import pathlib

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from mrcnn import utils
from mrcnn.model import log

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
        masks_neut = glob.glob(os.path.join(dataset_dir, "neutrophils", "*.png"))
        masks_epit = glob.glob(os.path.join(dataset_dir, "epithelium", "*.png"))

        for image_path, mask_neut_path, mask_epit_path in zip(images, masks_neut, masks_epit):

            image_path = pathlib.Path(image_path)
            mask_neut_path = pathlib.Path(mask_neut_path)
            mask_epit_path = pathlib.Path(mask_epit_path)

            print(image_path)
            print(mask_neut_path)
            print(mask_epit_path)

            #: ファイル名対応の検証
            assert image_path.name == mask_neut_path.name == mask_epit_path.name, 'データセット名不一致'

            image = Image.open(image_path)
            height = image.size[0]
            width = image.size[1]

            #: サイズ一致検証と画像ファイルを正常に開けるかの検証
            mask_neut = Image.open(mask_neut_path)
            mask_epit = Image.open(mask_epit_path)
            assert image.size == mask_neut.size == mask_epit.size, "サイズ不一致"

            self.add_image(
                'cell_dataset',
                path=image_path,
                image_id=image_path.stem,
                mask_path=(mask_neut_path, mask_epit_path),
                width=width, height=height,
                any_keyword_is_OK="add_imageメソッドは任意のキーワードで好きな情報を保持させることができる")

    def load_mask(self, image_id):
        """マスクデータとクラスidを生成する
        """
        image_info = self.image_info[image_id]
        mask_neut_path, mask_epit_path = image_info['mask_path']

        print(image_info)
        print(mask_neut_path, mask_epit_path)

        mask_neut, cls_idxs_neut = blob_detection(str(mask_neut_path), class_id=1)
        mask_epit, cls_idxs_epit = blob_detection(str(mask_epit_path), class_id=2)

        if mask_neut is not None and mask_epit is not None:
            mask = np.concatenate([mask_neut, mask_epit], axis=2)
            cls_idxs = np.concatenate([cls_idxs_neut, cls_idxs_epit])
        elif mask_neut is not None and mask_epit is None:
            mask = mask_neut
            cls_idxs = cls_idxs_neut
        elif mask_neut is None and mask_epit is not None:
            mask = mask_epit
            cls_idxs = cls_idxs_epit
        else:
            raise Exception("画像内に対象オブジェクトが一つも存在しない")

        return mask, cls_idxs

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == 'cell_dataset':
            return info
        else:
            super(self.__class__, self).image_reference(image_id)


def blob_detection(mask_path, class_id: int):
    """
        opencvのブロブ検出関数の戻り値が微妙に変わってる気がするのでダウングレード推奨
        最新はcv2.__version == 4.5.1なので pip install -U opencv-python==3.4.13

        N個の物体が入っている一枚の二値画像(height, width, 1)から,
        opencvのブロブ検出関数を利用して (height, widht, N)のNチャネル画像を生成する
        各チャネルが１つの物体のマスク情報となっている

        cls_idxsは各チャネルのマスク情報が示す物体のクラスidのリストとなっている
        ※もとのブログではすべての物体がcell(id=1)だったのでcls_idxsはnp.ones(N)でよかった
    """
    mask = cv2.imread(mask_path, 0)
    _, mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)

    tmp = cv2.connectedComponentsWithStats(mask)
    data = copy.deepcopy(tmp[1])

    #: 可視化するとブロブ検出関数の意味がよくわかる
    #: plt.imshow(data)
    #: plt.show()

    labels = []
    for label in np.unique(data):
        #: ラベルID==0は背景
        if label == 0:
            continue
        else:
            labels.append(label)

    if len(labels) == 0:
        #: 対象オブジェクトがない場合はNone
        return None, None
    else:
        mask = np.zeros((mask.shape)+(len(labels),), dtype=np.uint8)
        for n, label in enumerate(labels):
            mask[:, :, n] = np.uint8(data == label)
        cls_idxs = np.ones([mask.shape[-1]], dtype=np.int32) * class_id

        return mask, cls_idxs


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    TRAIN_DATASET = os.path.join('dataset', "train")
    dataset_train = ShapesDataset()
    dataset_train.load_dataset(TRAIN_DATASET)
    dataset_train.prepare()

    print()
    print("===="*10)
    print()

    from mrcnn import visualize
    image_ids = dataset_train.image_ids
    for TRAIN_DATASET in image_ids:
        image = dataset_train.load_image(TRAIN_DATASET)
        mask, class_ids = dataset_train.load_mask(TRAIN_DATASET)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
