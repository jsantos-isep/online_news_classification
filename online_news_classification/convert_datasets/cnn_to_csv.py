import logging
import time

import online_news_classification_lib.functions as functions
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def main():
    args = functions.setup_functions.get_arg_parser_to_csv().parse_args()
    start_time = functions.setup_functions.initialize("cnn_to_csv")
    dataset = functions.manage_datasets_functions.read_csv_dataset(
        filename=args.input, separator=","
    )
    dataset = dataset[dataset["Headline"] != ""]
    dataset = dataset[dataset["Description"] != ""]
    dataset["title"] = dataset["Headline"]
    dataset["category"] = dataset["Category"]
    dataset["abstract"] = dataset["Description"]
    dataset["date"] = dataset["Date published"]
    dataset["title_entities"] = pd.Series(dtype="object")
    dataset["abstract_entities"] = pd.Series(dtype="object")
    dataset["entities"] = pd.Series(dtype="object")
    dataset = dataset.drop(
        [
            "Index",
            "Headline",
            "Category",
            "Description",
            "Article text",
            "Url",
            "Second headline",
            "Date published",
            "Keywords",
        ],
        axis=1,
    )
    functions.manage_datasets_functions.save_dataset(dataset, args.output)
    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
