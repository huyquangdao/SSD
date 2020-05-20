import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.vgg import vgg16_bn

if torch.cuda.is_available():
    float_tensor = torch.cuda.FloatTensor
else:
    float_tensor = torch.FloatTensor


class BackBone(nn.Module):

    def __init__(self, modules, freeze = False):
      super(BackBone,self).__init__()

      self.backbone = nn.Sequential(*modules)
      if freeze:
          for param in self.backbone.parameters():
              param.requires_grad = False
    
    def forward(self, input):

        output = self.backbone(input)
        return output


class BottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride =1, padding = 1):

        super(BottleNeck,self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2,
                                 kernel_size=(1,1))
        
        self.conv3_3 = nn.Conv2d(in_channels=out_channels //2, out_channels=out_channels,
                                 kernel_size = (3,3), padding = padding, stride = stride)
    
    def forward(self, input):

        output = self.conv1_1(input)

        # print('input shape:',input.shape)

        # print('output shape:', output.shape)

        output = torch.relu(output)

        output = self.conv3_3(output)

        return output


class SSD(nn.Module):

    def __init__(self,image_size,n_classes, backbone = 'vgg16', freeze = False ):

        super(SSD,self).__init__()

        if backbone == 'vgg16':
            vgg16 = vgg16_bn(pretrained= True)
            modules = list(vgg16.children())[0][:30]
            self.backbone = BackBone(modules = modules, freeze = freeze)

        self.n_classes = n_classes
        self.image_size = image_size

        self.conv_out1 = nn.Conv2d(512, 4 * ( 4 + 1 + self.n_classes),(1,1))
        self.conv_forward1 = nn.Sequential(
                                  nn.Conv2d(512,1024,(3,3), padding=1, stride = 2),
                                  nn.Conv2d(1024,1024,(1,1))
                                 )

        self.conv_out2 = nn.Conv2d(1024, 6 *(4 + 1 + self.n_classes),(1,1))
        self.bottle_neck1 = BottleNeck(1024,512, 2)
        
        self.conv_out3 = nn.Conv2d(512,6 * (4 + 1 + self.n_classes),(1,1))
        self.bottle_neck2 = BottleNeck(512,256, 2)

        self.conv_out4 = nn.Conv2d(256,6 * (4 + 1 + self.n_classes),(1,1))
        self.bottle_neck3 = BottleNeck(256,256, 1, 0)

        self.conv_out5 = nn.Conv2d(256,4 * (4 + 1 + self.n_classes),(1,1))
        self.bottle_neck4 = BottleNeck(256,256, 1, 0)

        self.conv_out6 = nn.Conv2d(256,4 * (4 + 1 + self.n_classes),(1,1))

    
    def forward(self, image):

        output = self.backbone(image)
        
        out1 = self.conv_out1(output)

        output = self.conv_forward1(output)

        out2 = self.conv_out2(output)

        output = self.bottle_neck1(output)

        out3 = self.conv_out3(output)

        output = self.bottle_neck2(output)

        out4 = self.conv_out4(output)
        
        output = self.bottle_neck3(output)

        # print(output.shape)

        out5 = self.conv_out5(output)

        output = self.bottle_neck4(output)

        # print(output.shape)

        out6 = self.conv_out6(output)

        return out1, out2, out3, out4, out5, out6


class SSDLossLayer(nn.Module):

    def __init__(self, list_default_boxes, list_n_boxes, use_focal_loss = True, use_label_smooth = True, alpha = 1.):
        super(SSDLossLayer, self).__init__()
        self.list_default_boxes = list_default_boxes
        self.list_n_boxes = list_n_boxes
        self.alpha = alpha
        self.use_focal_loss = use_focal_loss
        self.use_label_smooth = use_label_smooth
        
    

    def forward(self, list_y_true, list_y_pred):

        y_true = torch.cat(list_y_true, dim=1)

        # for y in list_y_true:
        #     print(y.shape)

        default_boxes = torch.cat(self.list_default_boxes, dim = 0).to(self.device)

        default_boxes = default_boxes.unsqueeze(0)

        #default_boxes = [1,sum of box , 4]

        batch_size = y_true.shape[0]

        # for y in list_y_pred:
        #     print(y.shape)

        to_predict = []

        for i, n_boxes, in enumerate(self.list_n_boxes):

            map_shape = list_y_pred[i].shape[-1]

            temp = list_y_pred[i].clone()

            temp = temp.view(batch_size, -1, n_boxes, map_shape, map_shape )

            # print(temp.shape)

            temp =  temp.contiguous().view(batch_size, -1, n_boxes * map_shape * map_shape)

            temp = temp.permute(0,2,1)

            to_predict.append(temp)

        y_pred = torch.cat(to_predict, dim =1)

        true_boxes_xy = y_true[...,:2]
        true_boxes_wh = y_true[...,2:4]

        pred_boxes_xy = y_pred[...,:2]
        pred_boxes_wh = y_pred[...,2:4]

        pred_classes = y_pred[...,4:]
        true_classes = y_true[...,4:]

        positive_mask = y_true[...,4].unsqueeze(-1)
        negative_mask = 1 - positive_mask

        pred_no_object_cls =  y_pred[...,4:5]

        # print(torch.sum(y_true))

        true_boxes_wh = torch.where(true_boxes_wh == 0, x = torch.ones_like(true_boxes_wh).type(float_tensor), other = true_boxes_wh)

        true_boxes_xy = (true_boxes_xy - default_boxes[...,:2]) / default_boxes[...,2:4]
        true_boxes_wh = torch.log(torch.clamp(true_boxes_wh / default_boxes[...,2:4],1e-9,1e9))

        # print(torch.sum(true_boxes_wh))

        # print(true_boxes_xy.shape)

        xy_loss = positive_mask * torch.nn.functional.smooth_l1_loss(target = true_boxes_xy, input = pred_boxes_xy, reduction='none')
        wh_loss = positive_mask * torch.nn.functional.smooth_l1_loss(target = true_boxes_wh, input = pred_boxes_wh, reduction='none')

        N = torch.sum(positive_mask) 

        xy_loss = self.alpha * torch.sum(xy_loss) / N
        wh_loss = self.alpha * torch.sum(wh_loss) / N

        # print(pred_classes.shape)
        # print(true_classes.shape)

        cls_loss = positive_mask * torch.nn.functional.binary_cross_entropy_with_logits(input = pred_classes, target = true_classes, reduction = 'none')
        
        no_cls_loss = negative_mask * torch.nn.functional.binary_cross_entropy_with_logits(input = pred_no_object_cls, target = negative_mask, reduction = 'none')

        cls_loss = torch.sum(cls_loss) / N

        no_cls_loss = torch.sum(no_cls_loss) / N

        total_loss = xy_loss + wh_loss + cls_loss + no_cls_loss

        # print('xy_loss:{0}, wh_loss:{1}, cls_loss:{2}, no_cls_loss:{3}'.format(xy_loss,wh_loss,cls_loss,no_cls_loss))

        return total_loss, xy_loss, wh_loss, cls_loss, no_cls_loss