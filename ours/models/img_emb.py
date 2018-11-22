import torch.nn as nn
import torchvision.models as models

class ImageEmbedding(nn.Module):
    def __init__(self, feature_type='VGG16'):
        super(ImageEmbedding, self).__init__() # Must call super __init__()

        self.model_conv = None
        if feature_type == 'VGG16':
            self.model_conv = models.vgg16(pretrained=True)
            self.img_features = 512
        elif feature_type == 'Resnet152':
            self.model_conv = models.resnet152(pretrained=True)
            self.img_features = 2048
        else:
            print('Unsupported feature type: \'{}\''.format(feature_type))
            return None

        self.features = None
        if feature_type == 'VGG16':
            self.features = self.model_conv.features
        elif feature_type == 'Resnet152':
            self.features = nn.Sequential(*list(self.model_conv.children())[: -2])

        # Freeze all layers of the Conv Model
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, images):
        images = images.float()
        image_feats = self.features(images)
        
        return image_feats