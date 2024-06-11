import logging
import time

import online_news_classification_lib.functions as functions
from dotenv import load_dotenv

load_dotenv()


def main():
    args = functions.setup_functions.get_arg_parser_to_csv().parse_args()
    start_time = functions.setup_functions.initialize("mind_to_csv")
    logging.info("Start converting MiND to CSV")
    columns = [
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]
    dataset = functions.manage_datasets_functions.read_table_dataset(
        filename=args.input, columns=columns
    )
    dataset = dataset.dropna()
    dataset = dataset[dataset["title"] != ""]
    dataset = dataset[dataset["abstract"] != ""]
    dataset = dataset.drop(["url", "title_entities", "abstract_entities"], axis=1)
    functions.manage_datasets_functions.save_dataset(dataset, args.output)
    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
