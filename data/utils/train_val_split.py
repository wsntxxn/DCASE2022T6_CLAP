import json
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np


np.random.seed(1)


def split(args):
    input_json = args.audio_text_data
    val_percent = args.val_percent
    val_aid_ref = args.val_aid_ref
    train_json = args.output_train
    val_json = args.output_val

    data = json.load(open(input_json))["audios"]
    if val_aid_ref is None:
        train_data, val_data = train_test_split(data, test_size=val_percent)
    else:
        val_aids = [item["audio_id"] for item in json.load(
            open(val_aid_ref))["audios"]]
        val_aids = set(val_aids)
        train_data, val_data = [], []
        for item in data:
            if item["audio_id"] in val_aids:
                val_data.append(item)
            else:
                train_data.append(item)

    data_name = Path(input_json).stem
    if train_json is None:
        train_json = Path(input_json).with_name(data_name + "_train.json")
    if val_json is None:
        val_json = Path(input_json).with_name(data_name + "_val.json")
    json.dump({"audios": train_data}, open(train_json, "w"), indent=4)
    json.dump({"audios": val_data}, open(val_json, "w"), indent=4)
    print(f"{len(val_data)} items in validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_text_data", type=str)
    parser.add_argument("--val_percent", type=float, default=0.1, required=False)
    parser.add_argument("--val_aid_ref", type=str, default=None, required=False)
    parser.add_argument("--output_train", type=str, default=None, required=False)
    parser.add_argument("--output_val", type=str, default=None, required=False)

    args = parser.parse_args()
    split(args)
