import logging
import time

from dotenv import load_dotenv

from online_news_classification.functions import manage_datasets, setup

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
    dataset = dataset.drop(["headline", "short_description"], axis=1)
    manage_datasets.save_dataset(dataset, args.output)
    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
