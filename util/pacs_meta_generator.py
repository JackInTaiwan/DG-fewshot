import os
import json
import logging

from argparse import ArgumentParser

logger = logging.getLogger(__name__)



def generate_meta_file(domain_name, file_dir, output_dir):
    train_data = []

    for class_index, class_name in enumerate(sorted(os.listdir(os.path.join(file_dir, domain_name)))):
        train_data.append({"index": class_index, "paths": []})
        for fn in os.listdir(os.path.join(file_dir, domain_name, class_name)):
            train_data[class_index]["paths"].append(os.path.join(domain_name, class_name, fn))

    output_fp = os.path.join(output_dir, "{}.meta.json".format(domain_name))
    with open(output_fp, "w") as f:
        json.dump({"train": train_data}, f)

    return output_fp
        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--domain_name", action="store", type=str, required=True)
    parser.add_argument("-f", "--file_dir", action="store", type=str, required=True)
    parser.add_argument("-o", "--output_dir", action="store", type=str, required=True)

    args = parser.parse_args()
    domain_name, file_dir, output_dir = args.domain_name, args.file_dir, args.output_dir

    logging.info("| Start generating meta json file ...")
    output_fp = generate_meta_file(domain_name, file_dir, output_dir)
    logging.info("| Finish generating meta json file and save it in '{}' ...".format(output_fp))