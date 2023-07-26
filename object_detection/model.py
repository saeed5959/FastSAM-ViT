import torch
from torch import nn

from fastsam import FastSAM
from vit.model import Vit
from core.settings import model_config, train_config


class FastSamPredict(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = FastSAM(model_path)


    def forward(self, input):
        
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
        patch_all = self.bbox_to_patch(bbox_all)

        return patch_all
    
    
    def mask_to_bbox(self, masks):

        bbox_sam = torch.zeros((masks.size()[0],4))
        for count,mask in enumerate(masks):
            indexs = torch.where(mask==1)
            x1, y1, x2, y2 = min(indexs[1]), min(indexs[0]), max(indexs[1]), max(indexs[0])
            #remove small segments
            if x1 > m:
                bbox_sam[count,:] = torch.tensor([x1, y1, x2, y2])

        #combine segments
        for bbox in bbox_sam:
            bbox_all = combine

        return bbox_all
    
    def bbox_to_patch(self, bbox_all):

        return patch_all
    

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
        self.vit_model = VitFeature(vit_model_path)
        
        self.linear = nn.Linear(4,80)

    def forward(self, input, patch_all):
        with torch.no_grad():
            feature_vectors = self.vit_model(input)

        for patch in patch_all:
            patch_features = self.ROI_align(feature_vectors, patch)
            output = self.linear(patch_features)

        return output
    
    def ROI_align(self, feature_vectors, patch):


        return patch_features

