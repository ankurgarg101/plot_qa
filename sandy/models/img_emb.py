import torch.nn as nn
import torchvision.models as models

class ImageEmbedding(nn.Module):
    def __init__(self, hidden_size, feature_type='VGG16'):
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

        self.hidden_size = hidden_size
        self.linear = nn.Linear(self.img_features, self.hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, images):
        images = images.float()
        image_feats = self.features(images)

        image_feats = image_feats.permute(0, 2, 3, 1)
        image_feats_final = self.tanh(self.linear(image_feats))

        # input: [batch_size, 512, 14, 14]

        # intermed = self.linear(input.view(-1,self.img_features)).view(
        #                             -1, 196, self.hidden_size)
        return image_feats_final