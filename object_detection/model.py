import torch
from torch import nn

from fastsam import FastSAM
from core.settings import model_config, train_config


class FastSamPredict(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = FastSAM(model_path)


    def predict(self, input):
        
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

        bbox = self.mask_bbox(masks)

        return bbox
    
    
    def mask_bbox(self, masks):

        bbox = torch.zeros((masks.size()[0],4))
        for count,mask in enumerate(masks):
            indexs = torch.where(mask==1)
            x1, y1, x2, y2 = min(indexs[1]), min(indexs[0]), max(indexs[1]), max(indexs[0])
            bbox[count,:] = torch.tensor([x1, y1, x2, y2])

        return bbox