import logging
import time

import online_news_classification_lib.functions as functions
from dotenv import load_dotenv

load_dotenv()


def main():
    args = functions.setup_functions.get_arg_parser_to_csv().parse_args()
    start_time = functions.setup_functions.initialize("ag_news_to_csv")
    logging.info("Start converting AG News to CSV")
    dataset = functions.manage_datasets_functions.read_csv_dataset(
        filename=args.input_file, separator=","
    )
    dataset = dataset[dataset["Title"] != ""]
    dataset = dataset[dataset["Description"] != ""]
    dataset["title"] = dataset["Title"]
    dataset["category"] = dataset["Category"]
    dataset["abstract"] = dataset["Description"]
    dataset = dataset.drop(["Title", "Category", "Description"], axis=1)
    functions.manage_datasets_functions.save_dataset(dataset, args.output_file)
    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
