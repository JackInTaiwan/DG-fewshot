import os
import re
import json
import logging
import random
import numpy as np

from argparse import ArgumentParser
from util.logging import logging_config

logger = logging.getLogger(__name__)
logging_config()

random.seed(1234)



def get_dataset(file_dir):
    dataset = {}
    for class_name in os.listdir(file_dir):
        dataset[class_name] = []
        for fn in os.listdir(os.path.join(file_dir, class_name)):
            if re.match(r".*\.ppm", fn) is not None:
                dataset[class_name].append(os.path.join(os.path.basename(file_dir), class_name, fn))
    return dataset



def generate_meta_file(file_dir, output_fp, way, shot, val_episode, val_query_num, is_metadataset):
    # Generate validation episode dataset following MetaDataset sampling rules
    if is_metadataset:
        dataset = get_dataset(file_dir)
        val_data = []

        for episode in range(val_episode):
            way = random.choice(range(5, min([50, len(dataset)])))
            query_num_per_class = 10

            selected_class_name_list = random.sample(dataset.keys(), way)
            alpha_list = [random.uniform(np.log10(0.5), np.log10(2)) for _ in range(way)]
            total_class_size_portion = sum([np.exp(alpha)*len(dataset[class_name]) for alpha, class_name in zip(alpha_list, selected_class_name_list)])

            beta = random.uniform(0, 1)
            support_set_size = min([
                500,
                sum([np.ceil(beta * min([100, available_size-query_num_per_class])) for available_size in [len(dataset[class_name]) for class_name in selected_class_name_list]])
            ])
            
            support_data, query_data = [], []
            for class_index, class_name in enumerate(selected_class_name_list):
                shot = min([
                    1 + int((support_set_size - way) * np.exp(alpha_list[class_index]) * len(dataset[class_name]) / total_class_size_portion),
                    len(dataset[class_name]) - query_num_per_class
                ])
                selected_fn_list = random.sample(dataset[class_name], shot + query_num_per_class)
                selected_support_fn_list = selected_fn_list[:shot]
                selected_query_fn_list = selected_fn_list[shot:]
                
                support_data.append({"index": class_index, "paths": selected_support_fn_list})
                query_data.append({"index": class_index, "paths": selected_query_fn_list})
            
            val_data.append({
                "episode": episode,
                "support": support_data,
                "query": query_data
            })

    # Generate validation episode dataset following conventional N-way K-shot sampling rules
    else:
        pass
    
    output = {
        "validation": val_data
    }
    
    with open(output_fp, "w") as f:
        json.dump(output, f)

    return output_fp



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--file_dir", action="store", type=str, required=True)
    parser.add_argument("-o", "--output_fp", action="store", type=str, default="./data/meta/traffic_metadataset.meta.json")
    parser.add_argument("--metadataset", action="store_true")
    parser.add_argument("--way", action="store", type=int, default=5)
    parser.add_argument("--shot", action="store", type=int, default=5)
    parser.add_argument("--val_episode", action="store", type=int, default=600)
    parser.add_argument("--val_query_num", action="store", type=int, default=10)


    args = parser.parse_args()
    file_dir, output_fp = args.file_dir, args.output_fp
    is_metadataset = args.metadataset
    way, shot = args.way, args.shot
    val_episode, val_query_num = args.val_episode, args.val_query_num

    logging.info("| Start generating meta json file ...")
    output_fp = generate_meta_file(file_dir, output_fp, way, shot, val_episode, val_query_num, is_metadataset)
    logging.info("| Finish generating meta json file and save it in '{}' ...".format(output_fp))