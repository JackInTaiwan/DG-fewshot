import os
import re
import json
import logging
import random

from argparse import ArgumentParser
from util.logging import logging_config

logger = logging.getLogger(__name__)
logging_config()

random.seed(1234)



def get_dataset(file_dir):
    dataset = {}
    for fn in os.listdir(file_dir):
        if re.match(r".*\.jpg", fn) is not None:
            class_name = fn[:9]
            if class_name not in dataset:
                dataset[class_name] = []
            dataset[class_name].append(fn)
    
    logging.info("| Mini-imagenet Dataset contains {} classes.".format(len(dataset)))
    return dataset



def generate_meta_file(file_dir, output_dir, way, shot, test_episode, test_query_num, val_episode, val_query_num):
    dataset = get_dataset(file_dir)

    TEST_PERCENTAGE = 0.5

    test_class_name_pool = random.sample(dataset.keys(), int(len(dataset) * 0.7))
    val_class_name_pool = [class_name for class_name in dataset.keys() if class_name not in test_class_name_pool]

    # Generate test episode dataset
    test_data = []
    for episode in range(test_episode):
        selected_class_name_list = random.sample(test_class_name_pool, way)
        support_data, query_data = [], []

        for class_index, selected_class_name in enumerate(selected_class_name_list):
            selected_fn_list = random.sample(dataset[selected_class_name], shot + test_query_num)
            selected_support_fn_list = selected_fn_list[:shot]
            selected_query_fn_list = selected_fn_list[shot:]
            
            support_data.append({"index": class_index, "paths": selected_support_fn_list})
            query_data.append({"index": class_index, "paths": selected_query_fn_list})

        test_data.append({
            "episode": episode,
            "support": support_data,
            "query": query_data
        })

    # Generate validation episode dataset
    val_data = []
    for episode in range(val_episode):
        selected_class_name_list = random.sample(val_class_name_pool, way)
        support_data, query_data = [], []

        for class_index, selected_class_name in enumerate(selected_class_name_list):
            selected_fn_list = random.sample(dataset[selected_class_name], shot + val_query_num)
            selected_support_fn_list = selected_fn_list[:shot]
            selected_query_fn_list = selected_fn_list[shot:]
            
            support_data.append({"index": class_index, "paths": selected_support_fn_list})
            query_data.append({"index": class_index, "paths": selected_query_fn_list})

        val_data.append({
            "episode": episode,
            "support": support_data,
            "query": query_data
        })
    
    output = {
        "test": test_data,
        "validation": val_data
    }
    
    output_fp = os.path.join(output_dir, "miniimagenet.meta.json")
    with open(output_fp, "w") as f:
        json.dump(output, f)

    return output_fp



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--file_dir", action="store", type=str, required=True)
    parser.add_argument("-o", "--output_dir", action="store", type=str, required=True)
    parser.add_argument("--way", action="store", type=int, default=5)
    parser.add_argument("--shot", action="store", type=int, default=5)
    parser.add_argument("--test_episode", action="store", type=int, default=100)
    parser.add_argument("--test_query_num", action="store", type=int, default=50)
    parser.add_argument("--val_episode", action="store", type=int, default=50)
    parser.add_argument("--val_query_num", action="store", type=int, default=30)

    args = parser.parse_args()
    file_dir, output_dir = args.file_dir, args.output_dir
    way, shot = args.way, args.shot
    test_episode, test_query_num, val_episode, val_query_num = args.test_episode, args.test_query_num, args.val_episode, args.val_query_num

    logging.info("| Start generating meta json file ...")
    output_fp = generate_meta_file(file_dir, output_dir, way, shot, test_episode, test_query_num, val_episode, val_query_num)
    logging.info("| Finish generating meta json file and save it in '{}' ...".format(output_fp))