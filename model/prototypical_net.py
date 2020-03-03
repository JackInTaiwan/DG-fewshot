import copy
import torch
import numpy as np

from torch import nn



class PrototypicalNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.embedding_extractor = self.build_embedding_extractor()

        # dynamic
        self.kept_support_features = None   # (way, channel_size, FM_h, FM_w)


    def build_embedding_extractor(self):
        conv_block_channel_size = self.params["embedding_extractor.channel_size"]

        return nn.Sequential(
            # nn.InstanceNorm2d(3),
            nn.Conv2d(3, conv_block_channel_size, 3, padding=1),
            # nn.InstanceNorm2d(conv_block_channel_size),
            nn.BatchNorm2d(conv_block_channel_size),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(conv_block_channel_size, conv_block_channel_size, 3, padding=1),
            # nn.InstanceNorm2d(conv_block_channel_size),
            nn.BatchNorm2d(conv_block_channel_size),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(conv_block_channel_size, conv_block_channel_size, 3, padding=1),
            # nn.InstanceNorm2d(conv_block_channel_size),
            nn.BatchNorm2d(conv_block_channel_size),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(conv_block_channel_size, conv_block_channel_size, 3, padding=1),
            # nn.InstanceNorm2d(conv_block_channel_size),
            nn.BatchNorm2d(conv_block_channel_size),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    

    def forward(self, support_images, query_images):
        way, shot = support_images.size(0), support_images.size(1)
        
        support_images = support_images.view(-1, support_images.size(2), support_images.size(3), support_images.size(4))
        query_images = query_images.view(-1, query_images.size(2), query_images.size(3), query_images.size(4))
        
        embeddings = self.embedding_extractor(torch.cat([support_images, query_images], dim=0))
        support_embeddings = embeddings[:support_images.size(0)]
        support_embeddings = support_embeddings.view(way, shot, -1)
        support_embeddings = torch.mean(support_embeddings, dim=1)
        query_embeddings = embeddings[support_images.size(0):]
        repeated_support_embeddings = support_embeddings.repeat(query_embeddings.size(0), 1).unsqueeze(1)
        repeated_query_embeddings = query_embeddings.view(query_embeddings.size(0), -1).repeat(1, way)
        repeated_query_embeddings = repeated_query_embeddings.view(query_embeddings.size(0) * way, -1).unsqueeze(1)
        distances = torch.cdist(repeated_support_embeddings, repeated_query_embeddings)
        distances = distances.view(query_images.size(0), way)

        return distances


    def keep_support_features(self, support_images):
        # (way, shot, 3, h, w)
        way, shot = support_images.size(0), support_images.size(1)
        support_images = support_images.view(-1, support_images.size(2), support_images.size(3), support_images.size(4))
        with torch.no_grad():
            embeddings = self.embedding_extractor(support_images)
            embeddings = embeddings.view(way, shot, -1)
            embeddings = torch.mean(embeddings, dim=1).squeeze(dim=1)

        self.kept_support_features = embeddings


    def clean_support_features(self):
        self.kept_support_features = None


    def inference(self, query_images):
        """
        Note that self.kept_support_features must exist.
        """
        with torch.no_grad():
            support_embeddings = self.kept_support_features
            way = support_embeddings.size(0)

            query_embeddings = self.embedding_extractor(query_images)
            repeated_support_embeddings = support_embeddings.repeat(query_embeddings.size(0), 1).unsqueeze(1)
            repeated_query_embeddings = query_embeddings.view(query_embeddings.size(0), -1).repeat(1, way)
            repeated_query_embeddings = repeated_query_embeddings.view(query_embeddings.size(0) * way, -1).unsqueeze(1)
            
            distances = torch.cdist(repeated_support_embeddings, repeated_query_embeddings)
            distances = distances.view(query_images.size(0), way)
        
        return distances
