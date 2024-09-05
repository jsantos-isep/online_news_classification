import logging
import time

from dotenv import load_dotenv

from online_news_classification.functions import manage_datasets, setup

load_dotenv()


def main():
    args = setup.get_arg_parser_to_csv().parse_args()
    start_time = setup.initialize("ag_news_to_csv")
    logging.info("Start converting AG News to CSV")
    dataset = manage_datasets.read_csv_dataset(filename=args.input, separator=",")
    dataset = dataset[dataset["Title"] != ""]
    dataset = dataset[dataset["Description"] != ""]
    dataset["title"] = dataset["Title"]
    dataset["category"] = dataset["Category"]
    dataset["abstract"] = dataset["Description"]
    dataset = dataset.drop(["Title", "Category", "Description"], axis=1)
    manage_datasets.save_dataset(dataset, args.output)
    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
