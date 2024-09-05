import logging
import time

from dotenv import load_dotenv

from online_news_classification.functions import manage_datasets, setup

load_dotenv()


def main():
    args = setup.get_arg_parser_to_csv().parse_args()
    start_time = setup.initialize("cnn_to_csv")
    dataset = manage_datasets.read_csv_dataset(filename=args.input, separator=",")
    dataset = dataset[dataset["Headline"] != ""]
    dataset = dataset[dataset["Description"] != ""]
    dataset["title"] = dataset["Headline"]
    dataset["category"] = dataset["Category"]
    dataset["abstract"] = dataset["Description"]
    dataset["date"] = dataset["Date published"]
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
    manage_datasets.save_dataset(dataset, args.output)
    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
