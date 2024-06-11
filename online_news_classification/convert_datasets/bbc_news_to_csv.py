import logging
import time

import online_news_classification_lib.functions as functions
from dotenv import load_dotenv

load_dotenv()


def main():
    args = functions.setup_functions.get_arg_parser_to_csv().parse_args()
    start_time = functions.setup_functions.initialize("bbc_news_to_csv")
    logging.info("Start converting BBC News to CSV")
    dataset = functions.manage_datasets_functions.read_json_dataset(filename=args.input)
    dataset["abstract"] = dataset["short_description"]
    dataset["category"] = dataset["region"]
    dataset = dataset[dataset["title"] != ""]
    dataset = dataset[dataset["category"] != ""]
    dataset = dataset[dataset["abstract"] != ""]
    dataset = dataset.drop(["raw_content", "_id", "tags", "language", "region"], axis=1)
    functions.manage_datasets_functions.save_dataset(dataset, args.output)
    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
