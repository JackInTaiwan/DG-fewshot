import os
import copy
import json
import collections
import random
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, IterableDataset, DataLoader



class ClassDataset(IterableDataset):
    def __init__(self, index, items, input_image_size):
        self.class_index = index
        self.items = items
        self.item_pool = copy.deepcopy(self.items)
        self.input_image_size = input_image_size

        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(self.input_image_size)
        self.augmentation = transforms.Compose([
            transforms.RandomCrop(self.input_image_size),
        ])
        self.pytorch_pretrained_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )


    def __iter__(self):
        while True:
            if len(self.item_pool) == 0:
                self.item_pool = copy.deepcopy(self.items)
            
            image_fp = random.sample(self.item_pool, 1)[0]
            self.item_pool.remove(image_fp)

            image = Image.open(image_fp)
            image = self.image_preprocess(image)

            yield image


    def image_preprocess(self, image):
        image = image.convert('RGB')
        image = self.resize(image)
        image = self.to_tensor(image)
        image = self.pytorch_pretrained_normalize(image)

        return image



class FewShotDataLoader:
    def __init__(self, root_dir, meta_fp, way, shot, mode, input_image_size):
        self.root_dir = root_dir
        self.meta_fp = meta_fp
        self.way = way
        self.shot = shot
        self.mode = mode
        self.input_image_size = input_image_size

        self.class_dataset_dict = self.build_class_dataset_dict()


    def build_class_dataset_dict(self):
        with open(self.meta_fp) as f:
            data = json.load(f)[self.mode]

        build_class_dataset_dict = {}
        
        for class_data in data:
            class_index = class_data["index"]
            
            class_dataset_items = []
            for fp in class_data["paths"]:
                image_fp = os.path.join(self.root_dir, fp)
                class_dataset_items.append(image_fp)

            # create one class dataloader
            class_dataset = ClassDataset(
                index=class_index,
                items=class_dataset_items,
                input_image_size=self.input_image_size
            )
            build_class_dataset_dict[class_index] = iter(class_dataset)
        
        return build_class_dataset_dict
    

    def get_class_num(self):
        return len(self.class_dataset_dict)
    

    def get_images(self, class_index_list, shot):
        images = torch.empty((len(class_index_list), shot, 3, self.input_image_size[0], self.input_image_size[1]))
        
        for i, class_index in enumerate(class_index_list):
            dataset = self.class_dataset_dict[class_index]
            class_images = []
            for j in range(shot):
                image = next(dataset)
                images[i, j] = image

        return images



class CrossDomainSamplingDataLoader:
    def __init__(self, source_meta_list, root_dir, way, shot, batch_size, mode):
        self.root_dir = root_dir
        self.way = way
        self.shot = shot
        self.mode = mode
        self.batch_size = batch_size
        self.source_meta_list = source_meta_list

        # to be built
        self.class_num = None
        self.source_dataloader_pool = self.build_source_dataloader_pool()


    def build_source_dataloader_pool(self):
        source_dataloader_pool = []

        for meta_fp in self.source_meta_list:
            # FIXME
            # hard code input_image_size
            input_image_size = (64, 64)
            dataloader = FewShotDataLoader(self.root_dir, meta_fp, self.way, self.shot, self.mode, input_image_size)
            source_dataloader_pool.append(dataloader)

        # check and build self.class_num
        for dataloader in source_dataloader_pool:
            if self.class_num is not None and dataloader.get_class_num() != self.class_num:
                raise ValueError("Inconsistency of class number between domains!")
            else:
                self.class_num = dataloader.get_class_num()

        return source_dataloader_pool


    def __iter__(self):
        while True:
            # sample domains
            domain_indices = list(range(len(self.source_dataloader_pool)))
            selected_domain_indices = random.sample(domain_indices, 2)
            support_domain = self.source_dataloader_pool[selected_domain_indices[0]]
            query_domain = self.source_dataloader_pool[selected_domain_indices[1]]

            # sample classes
            selected_class_indices = random.sample(list(range(self.class_num)), self.way)
            support_images = support_domain.get_images(selected_class_indices, self.shot)   # (way, self.shot, 3, H, W)
            query_images = query_domain.get_images(selected_class_indices, self.batch_size) # (way, self.batch_size, 3, H, W)

            yield (support_images, query_images)



class CrossDomainSamplingEvalDataLoader:
    def __init__(self, mode, meta_fp, root_dir, batch_size, input_image_size=(64, 64)):
        super().__init__()
        self.mode = mode
        self.meta_fp = meta_fp
        self.root_dir = root_dir
        self.input_image_size = input_image_size
        self.batch_size = batch_size

        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(self.input_image_size)
        self.pytorch_pretrained_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # dynamic attrs
        self.data = None
        self.max_episode = 0
        self.current_episode = -1

        self.__init_data()


    def __init_data(self):
        with open(self.meta_fp) as f:
            self.data = json.load(f)[self.mode]
        self.max_episode = len(self.data)
    

    def get_support_images(self):
        episode_data = self.data[self.current_episode]
        support_images = []

        for class_data in episode_data["support"]:
            class_index, image_fp_list = class_data["index"], class_data["paths"]
            class_images = []

            for image_fp in image_fp_list:
                image = Image.open(os.path.join(self.root_dir, image_fp))
                image = self.__image_preprocess(image)
                image = image.unsqueeze(0)
                class_images.append(image)
            support_images.append(torch.cat(class_images).unsqueeze(0))

        support_images = torch.cat(support_images)

        return support_images
    

    def update_episode(self):
        self.current_episode += 1

        return self.current_episode < self.max_episode
    

    def reset_episode(self):
        self.current_episode = -1
    

    def __image_preprocess(self, image):
        image = image.convert('RGB')
        image = self.resize(image)
        image = self.to_tensor(image)
        image = self.pytorch_pretrained_normalize(image)

        return image


    def __iter__(self):
        """
        Return batched query images and corresponding labels.
        """
        episode_data = self.data[self.current_episode]
        image_data, lable_data = [], []

        for class_data in episode_data["query"]:
            class_index, image_fp_list = class_data["index"], class_data["paths"]
            for image_fp in image_fp_list:
                image = Image.open(os.path.join(self.root_dir, image_fp))
                image = self.__image_preprocess(image)
                image = image.unsqueeze(0)
                image_data.append(image)
                lable_data.append(class_index)
        
        image_data = torch.cat(image_data)
        lable_data = torch.tensor(lable_data, dtype=torch.float32)
        
        while len(image_data) > 0:
            images, labels = image_data[:self.batch_size], lable_data[:self.batch_size]
            yield (images, labels)
            image_data, lable_data = image_data[self.batch_size:], lable_data[self.batch_size:]



class IdenticalDomainSamplingDataLoader(CrossDomainSamplingDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    def __iter__(self):
        while True:
            # sample domains
            domain_indices = list(range(len(self.source_dataloader_pool)))
            selected_domain_indices = random.sample(domain_indices, 1)
            support_domain = self.source_dataloader_pool[selected_domain_indices[0]]
            query_domain = self.source_dataloader_pool[selected_domain_indices[0]]

            # sample classes
            selected_class_indices = random.sample(list(range(self.class_num)), self.way)
            support_images = support_domain.get_images(selected_class_indices, self.shot)   # (way, self.shot, 3, H, W)
            query_images = query_domain.get_images(selected_class_indices, 1)               # (way, 1, 3, H, W)

            yield (support_images, query_images)
            


class MixDomainSamplingDataLoader(CrossDomainSamplingDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def __iter__(self):
        while True:
            # sample domains
            domain_indices = list(range(len(self.source_dataloader_pool)))
            selected_domain_indices = random.choices(domain_indices, k=2)
            support_domain = self.source_dataloader_pool[selected_domain_indices[0]]
            query_domain = self.source_dataloader_pool[selected_domain_indices[1]]

            # sample classes
            selected_class_indices = random.sample(list(range(self.class_num)), self.way)
            support_images = support_domain.get_images(selected_class_indices, self.shot)   # (way, self.shot, 3, H, W)
            query_images = query_domain.get_images(selected_class_indices, self.batch_size) # (way, self.batch_size, 3, H, W)

            yield (support_images, query_images)