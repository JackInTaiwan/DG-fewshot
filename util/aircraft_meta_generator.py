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



def get_dataset(metadataset_split_file, aircraft_trainval_file, aircraft_test_file, aircraft_box_file):
    with open(metadataset_split_file) as f:
        split_data = json.load(f)
        test_split = split_data["test"]
    
    bbox_dataset = {}
    with open(aircraft_box_file) as f:
        for line in f.readlines():
            line = line.strip().split()
            image_id, bbox_1, bbox_2, bbox_3, bbox_4 = line[0], int(line[1]), int(line[2]), int(line[3]), int(line[4])
            bbox_dataset[image_id] = (bbox_1, bbox_2, bbox_3, bbox_4)

    with open(aircraft_trainval_file) as f:
        data_trainval = list(map(lambda x: [x.strip().split()[0], x.strip()[x.strip().index(" ")+1:]], f.readlines()))

    with open(aircraft_test_file) as f:
        data_test = list(map(lambda x: [x.strip().split()[0], x.strip()[x.strip().index(" ")+1:]], f.readlines()))
    
    data = data_trainval + data_test

    # collect valid data into a list
    test_dataset = {}
    test_aux_dataset = []

    for item in data:
        image_id, class_name = item[0], item[1]
        if class_name in test_split:
            image_fn = "{}.jpg".format(image_id)
            if class_name not in test_dataset:
                test_dataset[class_name] = [image_fn]
            else:
                test_dataset[class_name].append(image_fn)

            test_aux_dataset.append({
                "image_fn": image_fn,
                "bbox": bbox_dataset[image_id]
            })
    
    return test_dataset, test_aux_dataset


def generate_meta_file(metadataset_split_file, aircraft_trainval_file, aircraft_test_file, aircraft_bbox_file, images_dir, output_fp, way, shot, test_episode, test_query_num, val_episode, val_query_num, is_metadataset):
    test_dataset, test_aux_dataset = get_dataset(
        metadataset_split_file, aircraft_trainval_file, aircraft_test_file, aircraft_bbox_file
    )

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

    return output_fp, test_aux_dataset


def get_image_crop(file_path, bbox):
    try:
      img = Image.open(file_path)
    except:
      logging.warn("Failed to open image: {}".format(file_path))
      raise

    if img.mode != 'RGB':
      img = img.convert('RGB')

    if bbox is not None:
      img = img.crop(bbox)

    return img


def data_preprocess(aux_dataset, image_dir, target_image_dir):
    count_success, count_fail = 0, 0
    for item in aux_dataset:
        image_fn, bbox = item["image_fn"], item["bbox"]

        try:
            file_path = os.path.join(image_dir, image_fn)
            cropped_image = get_image_crop(file_path, bbox)
            cropped_image.save(os.path.join(target_image_dir, image_fn), "JPEG")
            count_success += 1
        except IOError as t:
            logging.warning('Image can not be opened and will be skipped.')
            count_fail += 1
            continue
        except ValueError:
            logging.warning('Image can not be cropped and will be skipped.')
            count_fail += 1
            continue

    return count_success, count_fail
        



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--metadataset_split_file", action="store", type=str, required=True)
    parser.add_argument("--aircraft_trainval_file", action="store", type=str, required=True)
    parser.add_argument("--aircraft_test_file", action="store", type=str, required=True)
    parser.add_argument("--aircraft_bbox_file", action="store", type=str, required=False)
    parser.add_argument("--images_dir", action="store", type=str, required=True)
    parser.add_argument("--target_image_dir", action="store", type=str, default="./Aircraft/")
    
    parser.add_argument("--metadataset", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    
    parser.add_argument("-o", "--output_fp", action="store", type=str, default="./data/meta/aircraft_metadataset.meta.json")
    parser.add_argument("--way", action="store", type=int, default=5)
    parser.add_argument("--shot", action="store", type=int, default=5)
    parser.add_argument("--test_episode", action="store", type=int, default=600)
    parser.add_argument("--test_query_num", action="store", type=int, default=50)
    parser.add_argument("--val_episode", action="store", type=int, default=600)
    parser.add_argument("--val_query_num", action="store", type=int, default=50)

    args = parser.parse_args()
    metadataset_split_file, aircraft_trainval_file, aircraft_test_file, aircraft_bbox_file, images_dir, target_image_dir, output_fp =\
        args.metadataset_split_file, args.aircraft_trainval_file, args.aircraft_test_file, args.aircraft_bbox_file, args.images_dir, args.target_image_dir, args.output_fp
    is_metadataset, do_preprocess = args.metadataset, args.preprocess
    way, shot = args.way, args.shot
    test_episode, test_query_num, val_episode, val_query_num = args.test_episode, args.test_query_num, args.val_episode, args.val_query_num

    ### Generate meta json file
    logging.info("| Start generating meta json file ...")
    output_fp, test_aux_dataset = generate_meta_file(metadataset_split_file, aircraft_trainval_file, aircraft_test_file, aircraft_bbox_file, images_dir, output_fp, way, shot, test_episode, test_query_num, val_episode, val_query_num, is_metadataset)
    logging.info("| Finish generating meta json file and save it in '{}' !!".format(output_fp))

    ### Preprocess raw data and dump them
    if do_preprocess:
        logging.info("| Start preprocess data ...")
        count_success, count_fail = data_preprocess(test_aux_dataset, images_dir, target_image_dir)
        logging.info("| Finish preprocessing data and save in '{}' !!".format(target_image_dir))
        logging.info("| Success: {} images | Fail: {} images".format(count_success, count_fail))
    