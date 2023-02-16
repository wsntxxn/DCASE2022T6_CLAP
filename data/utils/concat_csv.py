#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import glob
import argparse
import logging

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("input_csvs", type=str, nargs="+")
parser.add_argument("output_csv", type=str)
parser.add_argument("--sep", type=str, default="\t")
parser.add_argument("--float_format", type=str, default="%.3f")

args = parser.parse_args()

logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=logfmt)

output_dfs = []
for fname in args.input_csvs:
    if Path(fname).is_file():
        df = pd.read_csv(fname, sep=args.sep)
        output_dfs.append(df)
    elif "*" in fname:
        for f in glob.glob(fname):
            df = pd.read_csv(f, sep=args.sep)
            output_dfs.append(df)
    else:
        raise Exception(f"cannot recognize input {fname}")

output_df = pd.concat(output_dfs, join="inner")
output_df.reset_index(inplace=True, drop=True)
output_df.to_csv(args.output_csv, sep=args.sep, index=False, float_format=args.float_format)
logging.info("new csv has " + str(output_df.shape[0]) + " items")
