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



def get_dataset(metadataset_split_file, mscoco_annotation_train_file):
    with open(metadataset_split_file) as f:
        split_data = json.load(f)
        test_split = split_data["test"]
    
    with open(mscoco_annotation_train_file) as f:
        data = json.load(f)
        annotations = data["annotations"]
        categories = data["categories"]
        images = data["images"]

    # refactor and build up category dict
    categories_dict = dict(map(lambda x: (x["id"], x["name"]), categories))

    # refactor and build up image file name dict
    image_fn_dict = dict(map(lambda x: (x["id"], x["file_name"]), images))

    # collect valid data into a list
    val_dataset, test_dataset = {}, {}
    test_aux_dataset = []

    for annotation in annotations:
        _id, image_id, bbox, category_id = annotation["id"], annotation["image_id"], annotation["bbox"], annotation["category_id"]
        categorie_name = categories_dict[category_id]
        if categorie_name in test_split:
            image_fn = image_fn_dict[image_id]
            target_file_name = "{}.jpg".format(_id)
            if categorie_name not in test_dataset:
                test_dataset[categorie_name] = [target_file_name]
            else:
                test_dataset[categorie_name].append(target_file_name)

            test_aux_dataset.append({
                "_id": _id,
                "target_file_name": target_file_name,
                "image_fn": image_fn,
                "bbox": bbox,
                "category_id": category_id
            })
    
    return val_dataset, test_dataset, test_aux_dataset


def generate_meta_file(metadataset_split_file, mscoco_annotation_train_file, images_dir, output_fp, way, shot, test_episode, test_query_num, val_episode, val_query_num, is_metadataset):
    val_dataset, test_dataset, test_aux_dataset = get_dataset(metadataset_split_file, mscoco_annotation_train_file)

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


def get_image_crop_and_class_id(file_path, bbox, scale_ratio=1.2):
    # with tf.io.gfile.GFile(image_path, 'rb') as f:
    #     # The image shape is [?, ?, 3] and the type is uint8.
    image = Image.open(file_path)
    image = image.convert(mode='RGB')
    image_w, image_h = image.size

    def scale_box(bbox, scale_ratio):
        x, y, w, h = bbox
        x = x - 0.5 * w * (scale_ratio - 1.0)
        y = y - 0.5 * h * (scale_ratio - 1.0)
        w = w * scale_ratio
        h = h * scale_ratio
        return [x, y, w, h]

    x, y, w, h = scale_box(bbox, scale_ratio)
    # Convert half-integer to full-integer representation.
    # The Python Imaging Library uses a Cartesian pixel coordinate system,
    # with (0,0) in the upper left corner. Note that the coordinates refer
    # to the implied pixel corners; the centre of a pixel addressed as
    # (0, 0) actually lies at (0.5, 0.5). Since COCO uses the later
    # convention and we use PIL to crop the image, we need to convert from
    # half-integer to full-integer representation.
    xmin = max(int(round(x - 0.5)), 0)
    ymin = max(int(round(y - 0.5)), 0)
    xmax = min(int(round(x + w - 0.5)) + 1, image_w)
    ymax = min(int(round(y + h - 0.5)) + 1, image_h)

    image_crop = image.crop((xmin, ymin, xmax, ymax))
    crop_width, crop_height = image_crop.size

    if crop_width <= 0 or crop_height <= 0:
        raise ValueError('crops are not valid.')

    return image_crop


def data_preprocess(aux_dataset, image_dir, target_image_dir):
    count_success, count_fail = 0, 0
    for item in aux_dataset:
        _id, target_file_name, image_fn, bbox, category_id = item["_id"], item["target_file_name"], item["image_fn"], item["bbox"], item["category_id"]

        try:
            file_path = os.path.join(image_dir, image_fn)
            cropped_image = get_image_crop_and_class_id(file_path, bbox)
            cropped_image.save(os.path.join(target_image_dir, target_file_name), "JPEG")
            count_success += 1
        except IOError:
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
    parser.add_argument("--mscoco_annotation_train_file", action="store", type=str, required=True)
    parser.add_argument("--images_dir", action="store", type=str, required=True)
    parser.add_argument("--target_image_dir", action="store", type=str, default="./MSCOCO/")
    
    parser.add_argument("--metadataset", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    
    parser.add_argument("-o", "--output_fp", action="store", type=str, default="./data/meta/mscoco_metadataset.meta.json")
    parser.add_argument("--way", action="store", type=int, default=5)
    parser.add_argument("--shot", action="store", type=int, default=5)
    parser.add_argument("--test_episode", action="store", type=int, default=600)
    parser.add_argument("--test_query_num", action="store", type=int, default=50)
    parser.add_argument("--val_episode", action="store", type=int, default=600)
    parser.add_argument("--val_query_num", action="store", type=int, default=50)

    args = parser.parse_args()
    metadataset_split_file, mscoco_annotation_train_file, images_dir, target_image_dir, output_fp = args.metadataset_split_file, args.mscoco_annotation_train_file, args.images_dir, args.target_image_dir, args.output_fp
    is_metadataset, do_preprocess = args.metadataset, args.preprocess
    way, shot = args.way, args.shot
    test_episode, test_query_num, val_episode, val_query_num = args.test_episode, args.test_query_num, args.val_episode, args.val_query_num

    logging.info("| Start generating meta json file ...")
    output_fp, test_aux_dataset = generate_meta_file(metadataset_split_file, mscoco_annotation_train_file, images_dir, output_fp, way, shot, test_episode, test_query_num, val_episode, val_query_num, is_metadataset)
    logging.info("| Finish generating meta json file and save it in '{}' !!".format(output_fp))

    if do_preprocess:
        logging.info("| Start preprocess data ...")
        count_success, count_fail = data_preprocess(test_aux_dataset, images_dir, target_image_dir)
        logging.info("| Finish preprocessing data and save in '{}' !!".format(target_image_dir))
        logging.info("| Success: {} images | Fail: {} images".format(count_success, count_fail))
    