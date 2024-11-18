import logging
import os
import time

import pandas as pd
from dotenv import load_dotenv
from natsort import natsorted

from online_news_classification.functions import manage_datasets, setup

load_dotenv()


def extract_integer(filename):
    logging.info(filename)
    if filename.endswith(".csv"):
        return int(filename.split(".")[0].split("_")[1])
    else:
        return -1


def main():
    args = setup.get_arg_parser_merge().parse_args()
    start_time = setup.initialize("merge_files_" + args.dataset)
    input_dir = os.path.join(os.getcwd(), os.getenv("DATASETS_FOLDER") + args.input_dir)
    lines = []
    for filename in natsorted(os.listdir(input_dir)):
        if filename.endswith(".csv"):
            lines.append(pd.read_csv(os.path.join(input_dir, filename), delimiter=";"))
    df_res = pd.concat(lines, ignore_index=True)
    #df_res = df_res.drop(["Unnamed: 0"], axis=1)
    manage_datasets.save_dataset(df_res, args.output_file)
    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
