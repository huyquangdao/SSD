import torch
import cv2
import numpy as np
import math

def get_dataset_names(name_file_path):

    with open(name_file_path,'r') as f:
        names = {}
        for i,line in enumerate(f.readlines()):
            names[line.strip()] = i
        
        return names


def calculate_iou(box1, box2, x1y1x2y2=True):

    #box = [1,4]
    #anchors = [5776,4]

    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=1e-6
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-8)

    

    return iou

def letterbox_resize(image, new_image_size, interp = 0):

    ori_height, ori_width = image.shape[:2]

    new_width, new_height = new_image_size

    resize_ratio = min(new_height/ ori_height, new_width / ori_width)

    resize_w = int(resize_ratio * ori_width)
    
    resize_h = int(resize_ratio * ori_height)

    image_padded = np.full(shape=(new_height, new_width, 3),fill_value= 128).astype(np.uint8)

    image_resized = cv2.resize(image, (resize_w, resize_h), interp)

    dw = int((new_width - resize_w) / 2)
    dh = int((new_height - resize_h) / 2)

    image_padded[dh:dh + resize_h, dw:dw + resize_w] = image_resized

    return image_padded, resize_ratio, dw, dh


def resize_with_bboxes(image, bboxes, new_image_size, letterbox = True, interp = 0):

    if letterbox:

        image_resize, resize_ratio, dw, dh = letterbox_resize(image,new_image_size,interp)

        bboxes[:,[0,2]] = bboxes[:,[0,2]] * resize_ratio + dw
        bboxes[:,[1,3]] = bboxes[:,[1,3]] * resize_ratio + dh
    
    else:
        height, width = image.shape[:2]
        image_resize = cv2.resize(image, new_image_size, interp)
       
        h_ratio = new_image_size[1] / height
        w_ratio = new_image_size[0] / width

        bboxes[:,[0,2]] = bboxes[:,[0,2]] * w_ratio
        bboxes[:,[1,3]] = bboxes[:,[1,3]] * h_ratio
    
    return image_resize, bboxes


def resize_with_bboxes(image, bboxes, new_image_size, letterbox = True, interp = 0):

    if letterbox:

        image_resize, resize_ratio, dw, dh = letterbox_resize(image,new_image_size,interp)

        bboxes[:,[0,2]] = bboxes[:,[0,2]] * resize_ratio + dw
        bboxes[:,[1,3]] = bboxes[:,[1,3]] * resize_ratio + dh
    
    else:
        height, width = image.shape[:2]
        image_resize = cv2.resize(image, new_image_size, interp)
       
        h_ratio = new_image_size[1] / height
        w_ratio = new_image_size[0] / width

        bboxes[:,[0,2]] = bboxes[:,[0,2]] * w_ratio
        bboxes[:,[1,3]] = bboxes[:,[1,3]] * h_ratio
    
    return image_resize, bboxes


def plot_one_sample(image, boxes, labels):

    for label, box in list(zip(labels, boxes)):
        box = [int(t) for t in box]
        x1,y1,x2,y2 = box
        image = cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
    
    plt.imshow(image)
    plt.show()


def calculate_aspect_ratio(k, m = 6, s_min = 0.2, s_max = 0.9):
    s_k = s_min + ((s_max - s_min) / (m-1)) * (k-1)
    # print(s_k)
    return s_k 


def calculate_default_boxes(s_k, n_boxes = 6, aspect_ratio = [1., 2., 3., 1/2, 1/3, 1.] ):
    
    boxes = []

    for i in range(n_boxes):
        if i == n_boxes - 1:
            s_k = math.sqrt(s_k * s_k + 1)

        # print(i)

        w_a = s_k * math.sqrt(aspect_ratio[i])
        h_a = s_k / math.sqrt(aspect_ratio[i])

        boxes.append([w_a,h_a])
    
    return np.array(boxes)


def calculate_all_default_boxes(n_feature_maps = 6):

    n_feature_maps = n_feature_maps

    pair_map_boxes = {0:4,1:6,2:6,3:6,4:4,5:4}

    pair_map_aspect_ratio = {0:[1.,2.,1/2,1.],
                            1:[1.,2.,3.,1/2,1/3,1.],
                            2:[1.,2.,3.,1/2,1/3,1.],
                            3:[1.,2.,3.,1/2,1/3,1.],
                            4:[1.,2.,1/2,1.],
                            5:[1.,2.,1/2,1.]}
    
    # pair_map_aspect_ratio = {0:[1.,2.,1/2],
    #                         1:[1.,2.,3.,1/2,1/3],
    #                         2:[1.,2.,3.,1/2,1/3],
    #                         3:[1.,2.,3.,1/2,1/3],
    #                         4:[1.,2.,1/2],
    #                         5:[1.,2.,1/2]}

    pair_map_shape = {0:37,1:19,2:10,3:5,4:3,5:1}
    
    list_default_boxes =  []

    for i in range(n_feature_maps):
        n_boxes = pair_map_boxes[i]
        aspect_ratio = pair_map_aspect_ratio[i]
        s_k = calculate_aspect_ratio(i+1)

        # print(s_k)
        default_boxes = calculate_default_boxes(s_k = s_k, 
                                                n_boxes = n_boxes, 
                                                aspect_ratio = aspect_ratio)
        
        default_boxes = torch.FloatTensor(default_boxes)
        
        map_shape = pair_map_shape[i]        
        
        grid_size = np.arange(0,map_shape)

        a,b = np.meshgrid(grid_size, grid_size)

        x_offset = torch.FloatTensor(a).view(-1,1)
        y_offset = torch.FloatTensor(b).view(-1,1)

        x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,n_boxes).view( map_shape,  map_shape, n_boxes ,2)

        x_y_offset = (x_y_offset + 0.5)

        #xy_offset = [grid_size, grid_size, 2]
        #default_boxes = [N, 2]

        # x_y_offset = x_y_offset.unsqueeze(2).repeat(1,1,n_boxes,1)

        #xy_offset = [gird_size, grid_size, N, 2]

        default_boxes = default_boxes.unsqueeze(0).unsqueeze(0).repeat( map_shape,  map_shape, 1, 1)

        #default_boxes = [grid_size, grid_size, N, 2]

        default_boxes = torch.cat([x_y_offset,default_boxes],dim=-1)

        default_boxes = default_boxes / map_shape

        default_boxes = default_boxes.view(-1,4)

        # default_boxes = default_boxes 
        
        list_default_boxes.append(default_boxes)
    
    return list_default_boxes


def build_ground_truth(boxes, labels, image_size, n_classes, list_default_boxes, iou_thresh = 0.5):

    y_true = []

    y_true_38 = torch.zeros(size=(37 * 37 * 4, 4 + 1 + 1))
    y_true_19 = torch.zeros(size=(19 * 19 * 6, 4 + 1 + 1))
    y_true_10 = torch.zeros(size=(10 * 10 *6, 4 + 1 + 1))
    y_true_5 = torch.zeros(size=(5 * 5 * 6, 4 + 1 + 1))
    y_true_3 = torch.zeros(size=(3 * 3 * 4,4 + 1 + 1))
    y_true_1 = torch.zeros(size=(1 * 1 * 4, 4 + 1 + 1))

    y_true = [y_true_38, y_true_19, y_true_10, y_true_5, y_true_3, y_true_1]

    for i,box in enumerate(boxes):

        best_idx = None
        best_iou = -1
        best_mask = None

        n_box = None

        box = torch.FloatTensor(box).unsqueeze(0)

        temp1 = box.clone()

        box_centers = (box[:,0:2] + box[:,2:4]) /2
        box_sizes = (box[:,2:4] - box[:,0:2])

        box[:,0:2] = box_centers
        box[:,2:4] = box_sizes

        box = box / image_size

        for j, (y, default_boxes) in enumerate(list(zip(y_true, list_default_boxes))):

            map_shape = y.shape[0]

            # print(default_boxes.shape)

            # print(temp)

            # print(default_boxes[0])

            iou = calculate_iou(box, default_boxes)

            # print('iou:', iou.shape)

            #box = [1 , 4]

            #default_anchors = [shape,shape,M,4]

            #iou = [shape, shape, M, 1] 

            iou_mask = (iou > iou_thresh)

            # print('n matching box: ',torch.sum(iou_mask.type(torch.FloatTensor)))

            #iou_mask = [shape, shape, M]

            m_iou = torch.clamp(iou[iou_mask].mean(),min=0, max=1.)

            # print(m_iou)

            if m_iou > best_iou and True not in torch.isnan(m_iou):
                best_iou = m_iou
                best_idx = j
                best_mask = iou_mask
                # n_box = l
        
        if best_idx is not None:

            # print('iou_mask_shape:', best_mask.shape)
            # print('box shape:',box.shape)

            # print('y true ide',best_idx)

            # #[shape,shape, M]

            # print('y true shape:',y_true[best_idx][...,:,:4].shape)
            
            # print(''iou_mask.shape)

            # print('best_iou_mean_per_boxes: ', best_iou)
            # print('box coor:', temp1)

            # print('y_true shape: ',y_true[best_idx].shape)


            # print(y_true[best_idx][best_mask].shape)

            # map_shape = map_shape[best_idx]

            c = labels[i]

            # print(torch.sum(temp))

            y_true[best_idx][best_mask,:4] = box
            y_true[best_idx][best_mask,4] = 1.
            y_true[best_idx][best_mask,5] = c + 1

            # print(y_true[best_idx])

            # print('check here: ', torch.sum(y_true[best_idx]))

    return y_true


