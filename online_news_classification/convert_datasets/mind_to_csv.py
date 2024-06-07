import logging
import sys
import time

from dotenv import load_dotenv
from news_classification_lib.functions import (
    enrich_functions,
    manage_datasets_functions,
    setup_functions,
)

load_dotenv()


def main():
    args = setup_functions.get_arg_parser_to_csv().parse_args()
    start_time = setup_functions.initialize("mind_to_csv")
    logging.info("Start converting MiND to CSV")
    columns = [
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities"
    ]
    dataset = manage_datasets_functions.read_table_dataset(
        filename=args.input, columns=columns
    )
    dataset = dataset.dropna()
    dataset = dataset[dataset["title"] != ""]
    dataset = dataset[dataset["abstract"] != ""]
    dataset = dataset.drop(["url", "title_entities", "abstract_entities"], axis=1)
    manage_datasets_functions.save_dataset(dataset, args.output)
    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
