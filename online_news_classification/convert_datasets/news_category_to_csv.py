import logging
import time
import pandas as pd
from dotenv import load_dotenv
from news_classification_lib.functions import (
    manage_datasets_functions,
    setup_functions,
)

load_dotenv()


def main():
    args = setup_functions.get_arg_parser_to_csv().parse_args()
    start_time = setup_functions.initialize("news_category_to_csv")
    dataset = manage_datasets_functions.read_json_dataset(filename=args.input)
    dataset["title"] = dataset["headline"]
    dataset["abstract"] = dataset["short_description"]
    dataset = dataset[dataset["title"] != ""]
    dataset = dataset[dataset["abstract"] != ""]
    dataset = dataset.drop(["link"], axis=1)
    dataset["title_entities"] = pd.Series(dtype="object")
    dataset["abstract_entities"] = pd.Series(dtype="object")
    dataset = dataset.drop(["headline", "short_description"], axis=1)
    manage_datasets_functions.save_dataset(dataset, args.output)
    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
