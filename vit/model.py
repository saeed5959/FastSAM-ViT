from PIL import Image
import timm
import torch

img = Image.open('./dataset/000000000285.jpg').convert('RGB')

# model = timm.create_model('vit_base_patch16_clip_384.laion2b_ft_in12k_in1k', pretrained=True)
model = timm.create_model('maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k', pretrained=True, features_only=True)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1
# output_feature = model.forward_features(transforms(img).unsqueeze(0))
# top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
# print(model)
# print(output_feature.size())
# print(transforms(img).size())
# print(top5_class_indices)
# print(top5_probabilities)

for o in output:
    # print shape of each feature map in output
    # e.g.:
    #  torch.Size([1, 128, 192, 192])
    #  torch.Size([1, 128, 96, 96])
    #  torch.Size([1, 256, 48, 48])
    #  torch.Size([1, 512, 24, 24])
    #  torch.Size([1, 1024, 12, 12])

    print(o.shape)