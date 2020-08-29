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
    for class_name in os.listdir(file_dir):
        dataset[class_name] = []
        for fn in os.listdir(os.path.join(file_dir, class_name)):
            if re.match(r".*\.png", fn) is not None:
                dataset[class_name].append(os.path.join(os.path.basename(file_dir), class_name, fn))
    
    logging.info("| Alphabet '{}' Dataset contains {} classes.".format(file_dir, len(dataset)))
    return dataset



def generate_meta_file(file_dir, output_fp, way, shot, val_episode, val_query_num, is_metadataset):
    # Generate validation episode dataset following MetaDataset sampling rules
    if is_metadataset:
        val_data = []
        for episode in range(val_episode):
            selected_alphabet = random.sample(os.listdir(file_dir), k=1)[0]
            selected_alphabet_dp = os.path.join(file_dir, selected_alphabet)
            dataset = get_dataset(selected_alphabet_dp)
            
            way = random.choice(range(5, min([50, len(dataset)])))
            query_num_per_class = 10
            support_set_size = 10 * way
            
            selected_class_name_list = random.sample(os.listdir(selected_alphabet_dp), way)
            support_data, query_data = [], []
            dataset = get_dataset(selected_alphabet_dp)
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

    # Generate validation episode dataset following conventional N-way K-shot sampling rules
    else:
        val_data = []
        for episode in range(val_episode):
            selected_alphabet = random.sample(os.listdir(file_dir), k=1)[0]
            selected_alphabet_dp = os.path.join(file_dir, selected_alphabet)
            selected_class_name_list = random.sample(os.listdir(selected_alphabet_dp), way)
            support_data, query_data = [], []
            dataset = get_dataset(selected_alphabet_dp)
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
        "validation": val_data
    }
    
    with open(output_fp, "w") as f:
        json.dump(output, f)

    return output_fp



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--file_dir", action="store", type=str, required=True)
    parser.add_argument("-o", "--output_fp", action="store", type=str, default="./data/meta/omniglot_5_shot.meta.json")
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