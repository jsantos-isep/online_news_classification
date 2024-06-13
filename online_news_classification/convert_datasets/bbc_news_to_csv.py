import logging
import time

from dotenv import load_dotenv

from online_news_classification.functions import manage_datasets, setup

load_dotenv()


def main():
    args = setup.get_arg_parser_to_csv().parse_args()
    start_time = setup.initialize("bbc_news_to_csv")
    logging.info("Start converting BBC News to CSV")
    dataset = manage_datasets.read_json_dataset(filename=args.input)
    dataset["abstract"] = dataset["short_description"]
    dataset["category"] = dataset["region"]
    dataset = dataset[dataset["title"] != ""]
    dataset = dataset[dataset["category"] != ""]
    dataset = dataset[dataset["abstract"] != ""]
    dataset = dataset.drop(["raw_content", "_id", "tags", "language", "region"], axis=1)
    manage_datasets.save_dataset(dataset, args.output)
    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
