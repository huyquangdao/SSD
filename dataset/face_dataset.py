import matplotlib.pyplot as plt
# from utils.data_utils import resize_with_bbox, build_ground_truth, read_anchors
# from parser_data.face_parser import FaceParser
# from base.dataset import BaseDataset
import os
import cv2
from parser_data.face_parser import FaceParser
from base.dataset import BaseDataset
import numpy as np
from utils.data_utils import calculate_all_default_boxes, build_ground_truth, resize_with_bboxes


class FaceDataset(BaseDataset):

    def __init__(self,
                 type_name,
                 name_dir,
                 annotation_dir,
                 image_dir,
                 image_size,
                 letterbox=True,
                 is_train=True):

        self.parser = FaceParser(type_name, annotation_dir, name_dir)
        self.is_train = is_train
        self.image_dir = image_dir
        self.image_size = image_size
        self.is_train = is_train
        self.letterbox = letterbox
        self.list_default_boxes = calculate_all_default_boxes()

        # self.anchors = read_anchors(anchor_dir)
        self.dataset = self.parser.parse_dataset()

        print('num train samples: ',len(self.dataset))

        self.n_classes = len(self.parser.face_names)

    def __len__(self):
        return len(self.dataset)

    def __read_image(self, image_name):
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path)
        assert len(image.shape) == 3
        image = image[..., ::-1]
        return image

    def __getitem__(self, idx):

        file_name, labels, list_boxes = self.dataset[idx]['file_name'], self.dataset[
            idx]['labels'], self.dataset[idx]['list_all_boxes']

        # label_name = self.parser.face_names[labels]

        image = self.__read_image(file_name)

        image, boxes = resize_with_bboxes(image=image,
                                          bboxes=list_boxes,
                                          new_image_size = (self.image_size,self.image_size),
                                          letterbox=self.letterbox)
        
        image = image /255.

        image = image.astype(np.float32)
        
        # plot_one_sample(image,boxes,labels)


        assert image.shape == (self.image_size, self.image_size, 3)

        list_default_boxes = calculate_all_default_boxes()

        # print(list_default_boxes[-1])

        y_true = build_ground_truth(n_classes=self.n_classes,
                                    labels=labels,
                                    boxes=boxes,
                                    image_size=self.image_size,
                                    list_default_boxes = self.list_default_boxes)
        
        
        # print('in here:') 

        # print(torch.sum(y_true38))
        # print(torch.sum(y_true19))
        # print(torch.sum(y_true10))
        # print(torch.sum(y_true5))
        # print(torch.sum(y_true3))
        # print(torch.sum(y_true1))

        return image, y_true38, y_true19, y_true10, y_true5, y_true3, y_true1