import logging
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from functions import manage_datasets, setup

load_dotenv()


def main():
    args = setup.get_arg_parser_to_csv().parse_args()
    start_time = setup.initialize("news_category_to_csv")
    dataset = manage_datasets.read_json_dataset(filename=args.input)
    dataset["title"] = dataset["headline"]
    dataset["abstract"] = dataset["short_description"]
    dataset = dataset[dataset["title"] != ""]
    dataset = dataset[dataset["abstract"] != ""]
    dataset = dataset.drop(["link"], axis=1)
    dataset["title_entities"] = pd.Series(dtype="object")
    dataset["abstract_entities"] = pd.Series(dtype="object")
    dataset = dataset.drop(["headline", "short_description"], axis=1)
    manage_datasets.save_dataset(dataset, args.output)
    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
