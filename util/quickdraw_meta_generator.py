import os
import re
import json
import logging
import random

import numpy as np

from argparse import ArgumentParser
from util.logging import logging_config
from PIL import Image

logger = logging.getLogger(__name__)
logging_config()

random.seed(1234)



def get_dataset(metadataset_split_file, images_dir):
    with open(metadataset_split_file) as f:
        split_data = json.load(f)
        test_split = split_data["test"]

    # collect valid data into a list
    test_dataset = {}
    test_aux_dataset = []

    for class_name in test_split:
        file_name = "{}.npy".format(class_name)
        fp = os.path.join(images_dir, file_name)
        data = np.load(fp)
        test_dataset[class_name] = ["{}/{:0>6}.jpg".format(class_name, i) for i in range(len(data))]
    
    logging.info("Build up dataset of {} classes !".format(len(test_dataset)))
    
    return test_dataset


def generate_meta_file(metadataset_split_file, images_dir, output_fp, way, shot, test_episode, test_query_num, val_episode, val_query_num, is_metadataset):
    test_dataset = get_dataset(metadataset_split_file, images_dir)

    # Generate test episode dataset
    if is_metadataset:
        test_data = []
        for episode in range(val_episode):
            way = random.choice(range(5, min([50, len(test_dataset)])))
            query_num_per_class = 10

            selected_class_name_list = random.sample(test_dataset.keys(), way)
            alpha_list = [random.uniform(np.log10(0.5), np.log10(2)) for _ in range(way)]
            total_class_size_portion = sum([np.exp(alpha)*len(test_dataset[class_name]) for alpha, class_name in zip(alpha_list, selected_class_name_list)])

            beta = random.uniform(0, 1)
            support_set_size = min([
                500,
                sum([np.ceil(beta * min([100, available_size-query_num_per_class])) for available_size in [len(test_dataset[class_name]) for class_name in selected_class_name_list]])
            ])
            
            support_data, query_data = [], []
            for class_index, class_name in enumerate(selected_class_name_list):
                shot = min([
                    1 + int((support_set_size - way) * np.exp(alpha_list[class_index]) * len(test_dataset[class_name]) / total_class_size_portion),
                    len(test_dataset[class_name]) - query_num_per_class
                ])
                selected_fn_list = random.sample(test_dataset[class_name], shot + query_num_per_class)
                selected_support_fn_list = selected_fn_list[:shot]
                selected_query_fn_list = selected_fn_list[shot:]
                
                support_data.append({"index": class_index, "paths": selected_support_fn_list})
                query_data.append({"index": class_index, "paths": selected_query_fn_list})

            test_data.append({
                "episode": episode,
                "support": support_data,
                "query": query_data
            })
    
        output = {
            "validation": test_data,
        }
    
    with open(output_fp, "w") as f:
        json.dump(output, f)

    return output_fp, test_dataset


def data_preprocess(dataset, image_dir, target_image_dir):
    for class_name in dataset.keys():
        target_class_dir = os.path.join(target_image_dir, class_name)
        
        if not os.path.exists(target_class_dir): os.mkdir(target_class_dir)

        data = np.load(os.path.join(image_dir, "{}.npy".format(class_name)))
        for i, image in enumerate(data):
            target_fp = os.path.join(target_class_dir, "{:0>6}.jpg".format(i))
            image = image.reshape(int(image.shape[0]**0.5), int(image.shape[0]**0.5))
            image = Image.fromarray(image)
            image = image.convert("RGB")
            image.save(target_fp)

        logging.info("Done preprocess data of '{}' with {} images!".format(class_name, len(data)))



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--metadataset_split_file", action="store", type=str, required=True)
    parser.add_argument("--images_dir", action="store", type=str, required=True)
    parser.add_argument("--target_image_dir", action="store", type=str, default="./QuickDraw/images/")
    
    parser.add_argument("--metadataset", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    
    parser.add_argument("-o", "--output_fp", action="store", type=str, default="./data/meta/quickdraw_metadataset.meta.json")
    parser.add_argument("--way", action="store", type=int, default=5)
    parser.add_argument("--shot", action="store", type=int, default=5)
    parser.add_argument("--test_episode", action="store", type=int, default=600)
    parser.add_argument("--test_query_num", action="store", type=int, default=50)
    parser.add_argument("--val_episode", action="store", type=int, default=600)
    parser.add_argument("--val_query_num", action="store", type=int, default=50)

    args = parser.parse_args()
    metadataset_split_file, images_dir, target_image_dir, output_fp =\
        args.metadataset_split_file, args.images_dir, args.target_image_dir, args.output_fp
    is_metadataset, do_preprocess = args.metadataset, args.preprocess
    way, shot = args.way, args.shot
    test_episode, test_query_num, val_episode, val_query_num = args.test_episode, args.test_query_num, args.val_episode, args.val_query_num

    ### Generate meta json file
    logging.info("| Start generating meta json file ...")
    output_fp, dataset = generate_meta_file(metadataset_split_file, images_dir, output_fp, way, shot, test_episode, test_query_num, val_episode, val_query_num, is_metadataset)
    logging.info("| Finish generating meta json file and save it in '{}' !!".format(output_fp))

    ### Preprocess raw data and dump them
    if do_preprocess:
        logging.info("| Start preprocess data ...")
        data_preprocess(dataset, images_dir, target_image_dir)
        logging.info("| Finish preprocessing data and save in '{}' !!".format(target_image_dir))
    