import copy
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18



class RelationNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.embedding_extractor = self.build_embedding_extractor()
        self.relation_net = self.build_relation_net()

        # dynamic
        self.kept_support_features = None   # (way, channel_size, FM_h, FM_w)


    def build_embedding_extractor(self):
        if self.params["embedding_extractor.backbone"] == "block5":
            conv_block_channel_size = self.params["embedding_extractor.channel_size"]

            return nn.Sequential(
                nn.Conv2d(3, conv_block_channel_size, 3, padding=1),
                nn.BatchNorm2d(conv_block_channel_size),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(conv_block_channel_size, conv_block_channel_size, 3, padding=1),
                nn.BatchNorm2d(conv_block_channel_size),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(conv_block_channel_size, conv_block_channel_size, 3, padding=1),
                nn.BatchNorm2d(conv_block_channel_size),
                nn.ReLU(True),

                nn.Conv2d(conv_block_channel_size, conv_block_channel_size, 3, padding=1),
                nn.BatchNorm2d(conv_block_channel_size),
                nn.ReLU(True),

                nn.Conv2d(conv_block_channel_size, conv_block_channel_size, 3, padding=1),
                nn.BatchNorm2d(conv_block_channel_size),
                nn.ReLU(True),
            )

        if self.params["embedding_extractor.backbone"] == "block4":
            conv_block_channel_size = self.params["embedding_extractor.channel_size"]

            return nn.Sequential(
                nn.Conv2d(3, conv_block_channel_size, 3, padding=1),
                nn.BatchNorm2d(conv_block_channel_size),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(conv_block_channel_size, conv_block_channel_size, 3, padding=1),
                nn.BatchNorm2d(conv_block_channel_size),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(conv_block_channel_size, conv_block_channel_size, 3, padding=1),
                nn.BatchNorm2d(conv_block_channel_size),
                nn.ReLU(True),

                nn.Conv2d(conv_block_channel_size, conv_block_channel_size, 3, padding=1),
                nn.BatchNorm2d(conv_block_channel_size),
                nn.ReLU(True),
            )
        
        if self.params["embedding_extractor.backbone"] == "block3":
            conv_block_channel_size = self.params["embedding_extractor.channel_size"]

            return nn.Sequential(
                nn.Conv2d(3, conv_block_channel_size, 3, padding=1),
                nn.BatchNorm2d(conv_block_channel_size),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(conv_block_channel_size, conv_block_channel_size, 3, padding=1),
                nn.BatchNorm2d(conv_block_channel_size),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(conv_block_channel_size, conv_block_channel_size, 3, padding=1),
                nn.BatchNorm2d(conv_block_channel_size),
                nn.ReLU(True),
            )

        elif self.params["embedding_extractor.backbone"] == "resnet18":
            resnet = resnet18(pretrained=False)
            
            return nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4
            )

    
    def build_relation_net(self):
        conv_block_channel_size = self.params["embedding_extractor.channel_size"]

        return nn.Sequential(
            nn.Conv2d(conv_block_channel_size*2, conv_block_channel_size, 3, padding=1),
            nn.BatchNorm2d(conv_block_channel_size),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(conv_block_channel_size, conv_block_channel_size, 3, padding=1),
            nn.BatchNorm2d(conv_block_channel_size),
            nn.ReLU(True),
            nn.AdaptiveMaxPool2d((4, 4)),

            nn.Flatten(),
            nn.Linear(4*4*conv_block_channel_size, 8),
            nn.ReLU(True),

            nn.Linear(8, 1),
            nn.Sigmoid()
        )


    def forward(self, support_images, query_images):
        way, shot = support_images.size(0), support_images.size(1)
        
        support_images = support_images.view(-1, support_images.size(2), support_images.size(3), support_images.size(4))
        query_images = query_images.view(-1, query_images.size(2), query_images.size(3), query_images.size(4))
        
        embeddings = self.embedding_extractor(torch.cat([support_images, query_images], dim=0))
        support_embeddings = embeddings[:support_images.size(0)]
        support_embeddings = support_embeddings.view(way, shot, support_embeddings.size(1), support_embeddings.size(2), support_embeddings.size(3))
        support_embeddings = torch.sum(support_embeddings, dim=1).squeeze(dim=1)
        query_embeddings = embeddings[support_images.size(0):].squeeze(dim=1)
        repeated_support_embeddings = support_embeddings.repeat(query_embeddings.size(0), 1, 1, 1)
        repeated_query_embeddings = query_embeddings.repeat(1, way, 1, 1)
        repeated_query_embeddings = repeated_query_embeddings.view(query_embeddings.size(0) * way, query_embeddings.size(1), query_embeddings.size(2), query_embeddings.size(3))
        scores = self.relation_net(torch.cat([repeated_support_embeddings, repeated_query_embeddings], dim=1))
        scores = scores.view(query_images.size(0), way)

        return scores


    def keep_support_features(self, support_images):
        # (way, shot, 3, h, w)
        way, shot = support_images.size(0), support_images.size(1)
        support_images = support_images.view(-1, support_images.size(2), support_images.size(3), support_images.size(4))
        
        with torch.no_grad():
            embeddings = self.embedding_extractor(support_images)
            embeddings = embeddings.view(way, shot, embeddings.size(1), embeddings.size(2), embeddings.size(3))
            embeddings = torch.sum(embeddings, dim=1).squeeze(dim=1)

        self.kept_support_features = embeddings


    def clean_support_features(self):
        self.kept_support_features = None


    def inference(self, query_images):
        """
        Note that self.kept_support_features must exist.
        """
        with torch.no_grad():
            support_embeddings = self.kept_support_features
            query_embeddings = self.embedding_extractor(query_images)
            way = support_embeddings.size(0)

            repeated_support_embeddings = support_embeddings.repeat(query_embeddings.size(0), 1, 1, 1)
            repeated_query_embeddings = query_embeddings.repeat(1, way, 1, 1)
            repeated_query_embeddings = repeated_query_embeddings.view(query_embeddings.size(0) * way, query_embeddings.size(1), query_embeddings.size(2), query_embeddings.size(3))
            scores = self.relation_net(torch.cat([repeated_support_embeddings, repeated_query_embeddings], dim=1))
            scores = scores.view(query_images.size(0), way)
        
        return scores
