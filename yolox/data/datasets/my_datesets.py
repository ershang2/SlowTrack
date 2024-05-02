import cv2
import numpy as np
from pycocotools.coco import COCO

import os

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset

import glob
class StraDataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="train_half.json",
        name="train",
        img_size=(608, 1088),
        preproc=None,
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        print(data_dir)
        if os.path.isdir(data_dir):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            # self.files = sorted(glob.glob('%s/*.*' % path))
            # self.files = sorted(glob.glob('%s/*.*' % path), key=lambda x: int(os.path.splitext(x)[0]))
            self.files = sorted(glob.glob('%s/*.*' % data_dir), key=lambda x: os.path.splitext(os.path.basename(x))[0])
            self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.files))
        elif os.path.isfile(data_dir):
            self.files = [data_dir]
        self.files = self.files[:400]
        super().__init__(img_size)
        self.data_dir = data_dir
        self.json_file = json_file
        self.nF = len(self.files)  # number of image files

        # self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        # self.ids = self.coco.getImgIds()
        # self.class_ids = sorted(self.coco.getCatIds())
        # cats = self.coco.loadCats(self.coco.getCatIds())
        # self._classes = tuple([c["name"] for c in cats])
        # self.annotations = self._load_coco_annotations()
        self.name = name
        self.img_size = img_size
        self.preproc = preproc

    def __len__(self):
        return int(self.nF/2)  # number of files

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        frame_id = im_ann["frame_id"]
        video_id = im_ann["video_id"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = x1 + obj["bbox"][2]
            y2 = y1 + obj["bbox"][3]
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 6))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            res[ix, 5] = obj["track_id"]

        file_name = im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id_) + ".jpg"
        img_info = (height, width, frame_id, video_id, file_name)

        del im_ann, annotations

        return (res, img_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def pull_item(self, img_file):

        # load image and preprocess
        # img_file = os.path.join(
        #     self.data_dir, self.name, file_name
        # )
        img = cv2.imread(img_file)
        assert img is not None

        return img

    @Dataset.resize_getitem
    def __getitem__(self, idx):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            img_id (int): same as the input index. Used for evaluation.
        """
        idx_1 = int(idx*2 % self.nF)
        img_path = self.files[idx_1]
        img = self.pull_item(img_path)
        idx_2 = int((idx*2+1) % self.nF)
        img_path_2 = self.files[idx_2]
        img_2 = self.pull_item(img_path_2)

        if self.preproc is not None:
            img, target = self.preproc(img, self.input_dim)
            img_2, target = self.preproc(img_2, self.input_dim)
        return img,img_path,img_2,img_path_2
