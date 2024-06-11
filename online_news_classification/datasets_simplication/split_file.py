import logging
import os
import time

import pandas as pd
from dotenv import load_dotenv

from online_news_classification.lib.functions import setup

load_dotenv()


def main():
    args = setup.get_arg_parser_split().parse_args()
    start_time = setup.initialize("split_file_" + args.dataset)
    input_file = os.path.join(
        os.getcwd(), os.getenv("DATASETS_FOLDER") + args.input_file
    )
    output_file = os.path.join(os.getcwd(), os.getenv("DATASETS_FOLDER") + args.out_dir)

    with pd.read_csv(input_file, chunksize=args.num_lines, delimiter=";") as reader:
        for i, chunk in enumerate(reader):
            chunk = chunk.drop(["Unnamed: 0"], axis=1)
            chunk.to_csv(
                output_file + args.dataset + "_" + str(i) + ".csv", header=True, sep=";"
            )
    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
