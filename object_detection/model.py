import torch
from torch import nn

from fastsam import FastSAM
from vit.model import Vit
from core.settings import model_config, train_config


class FastSamPredict(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = FastSAM(model_path)

        self.height = model_config.height 
        self.width = model_config.width
        self.num_patch_w = model_config.num_patch_w
        self.num_patch_h = model_config.num_patch_h


    def forward(self, input):
        #taking segment mask result from fastsam : results consists of masks: every mask relate to an specific segment and object
        results = self.model(
            input,
            device=train_config.device,
            retina_masks=model_config.retina_masks,
            imgsz=model_config.imgsz,
            conf=model_config.conf,
            iou=model_config.iou    
            )

        masks = results[0].masks.data
        print(masks.size())

        bbox_all = self.mask_to_bbox(masks)

        return bbox_all
    
    
    def mask_to_bbox(self, masks):
        #convert segment mask to a bounding box
        bbox_sam = torch.zeros((masks.size()[0],4))
        for count,mask in enumerate(masks):
            indexs = torch.where(mask==1)
            #finding bbox coordinate : top-left(x1,y1)  bottom-right(x2,y2)
            x1, y1, x2, y2 = min(indexs[1]), min(indexs[0]), max(indexs[1]), max(indexs[0])
            #remove small segments
            if torch.sum(mask) > (self.height/self.num_patch_h)*(self.width/self.num_patch_w):
                bbox_sam[count,:] = torch.tensor([x1, y1, x2, y2])

        #combine segments
        for bbox in bbox_sam:
            bbox_all = combine

        return bbox_all

    

class VitFeature(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = Vit(model_path)
    
    def forward(self, input):
        feature_vectors = self.model(input)

        return feature_vectors
    

class FastSAMVit(nn.Module):
    def __init__(self, vit_model_path):
        super().__init__()
        self.height = model_config.height 
        self.width = model_config.width

        self.vit_model = VitFeature(vit_model_path)
        self.linear = nn.Linear(4,80)

    def forward(self, input, bbox_all):
        #freeze vit model without last layer of classification
        with torch.no_grad():
            feature_vectors = self.vit_model(input)

        for bbox in bbox_all:
            bbox_features = self.ROI_align(feature_vectors, bbox)
            output = self.linear(bbox_features)

        return output
    
    def ROI_align(self, feature_vectors, bbox):
        x1, y1, x2, y2 = bbox[0]/self.width, bbox[0]/self.height, bbox[1]/self.width, bbox[1]/self.height

        return bbox_features

