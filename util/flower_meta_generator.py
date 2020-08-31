import os
import re
import json
import logging
import random

import numpy as np

from argparse import ArgumentParser
from util.logging import logging_config
from scipy.io import loadmat

logger = logging.getLogger(__name__)
logging_config()

random.seed(1234)



def get_dataset(split_file, images_dir, labels_file):
    train_dataset, test_dataset = {}, {}
    with open(split_file) as f:
        split_data = json.load(f)
    test_split = split_data["test"]
    test_split = list(map(
        lambda x: int(x.split(".")[0]),
        test_split
    ))

    labels = loadmat(labels_file)["labels"][0]

    # build up test dataset
    for label, file_name in zip(labels, sorted(os.listdir(images_dir))):
        if int(label) in test_split:
            class_name = label
            file_path = os.path.join(images_dir, file_name)
            if class_name not in test_dataset:
                test_dataset[class_name] = [file_path]
            else:
                test_dataset[class_name].append(file_path)
    
    return train_dataset, test_dataset


def generate_meta_file(split_file, images_dir, labels_file, output_fp, way, shot, test_episode, test_query_num, val_episode, val_query_num, is_metadataset):
    train_dataset, test_dataset = get_dataset(split_file, images_dir, labels_file)
    
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

        # # Generate validation episode dataset
        # val_data = []
        # for episode in range(val_episode):
        #     selected_class_name_list = random.sample(val_class_name_pool, way)
        #     support_data, query_data = [], []

        #     for class_index, selected_class_name in enumerate(selected_class_name_list):
        #         selected_fn_list = random.sample(dataset[selected_class_name], shot + val_query_num)
        #         selected_support_fn_list = selected_fn_list[:shot]
        #         selected_query_fn_list = selected_fn_list[shot:]
                
        #         support_data.append({"index": class_index, "paths": selected_support_fn_list})
        #         query_data.append({"index": class_index, "paths": selected_query_fn_list})

        #     val_data.append({
        #         "episode": episode,
        #         "support": support_data,
        #         "query": query_data
        #     })
    
        output = {
            "validation": test_data,
            # "validation": val_data
        }
    
    with open(output_fp, "w") as f:
        json.dump(output, f)

    return output_fp



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--split_file", action="store", type=str, required=True)
    parser.add_argument("--images_dir", action="store", type=str, required=True)
    parser.add_argument("--labels_file", action="store", type=str, required=True)
    parser.add_argument("--metadataset", action="store_true")
    parser.add_argument("-o", "--output_fp", action="store", type=str, default="./data/meta/flower_metadataset.meta.json")
    parser.add_argument("--way", action="store", type=int, default=5)
    parser.add_argument("--shot", action="store", type=int, default=5)
    parser.add_argument("--test_episode", action="store", type=int, default=600)
    parser.add_argument("--test_query_num", action="store", type=int, default=50)
    parser.add_argument("--val_episode", action="store", type=int, default=600)
    parser.add_argument("--val_query_num", action="store", type=int, default=50)

    args = parser.parse_args()
    split_file, images_dir, labels_file, output_fp = args.split_file, args.images_dir, args.labels_file, args.output_fp
    is_metadataset = args.metadataset
    way, shot = args.way, args.shot
    test_episode, test_query_num, val_episode, val_query_num = args.test_episode, args.test_query_num, args.val_episode, args.val_query_num

    logging.info("| Start generating meta json file ...")
    output_fp = generate_meta_file(split_file, images_dir, labels_file, output_fp, way, shot, test_episode, test_query_num, val_episode, val_query_num, is_metadataset)
    logging.info("| Finish generating meta json file and save it in '{}' ...".format(output_fp))