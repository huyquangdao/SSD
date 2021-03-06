# from utils.data_utils import get_voc_names
import numpy as np
# from base.parser import BaseParser
from base.parser import BaseParser
from utils.data_utils import get_dataset_names


class FaceParser(BaseParser):

    def __init__(self, type_name, file_dir, name_dir):
        super().__init__(type_name, file_dir, name_dir)
        self.face_names = get_dataset_names(self.name_dir)

    def process_one(self, file):

        with open(file, 'r') as f:
            lines = f.readlines()

            list_labels = []
            list_boxes = []
            image_name = lines[0].strip()

            for line in lines[1:-2]:
                label, x_min, y_min, x_max, y_max = line.strip().split(' ')
                list_labels.append(self.face_names[label])
                list_boxes.append([float(x_min), float(
                    y_min), float(x_max), float(y_max)])

            return image_name, np.array(list_labels), np.array(list_boxes)
