import json
import logging
import os
import time
from datetime import datetime

import online_news_classification_lib.functions as functions
import requests
from dotenv import load_dotenv

load_dotenv()


def main():
    args = functions.setup_functions.get_arg_parser_get_dataset_from_api().parse_args()
    start_time = functions.setup_functions.initialize("get_dataset")
    API_KEY = os.getenv("NYTIMES_API_KEY")

    current_month = args.month
    current_year = args.year

    now_year = datetime.now().year

    while current_year <= now_year:
        if current_month <= 12:
            API_URL = (
                "https://api.nytimes.com/svc/archive/v1/"
                + str(current_year)
                + "/"
                + str(current_month)
                + ".json?api-key="
                + API_KEY
            )
            logging.info(API_URL)
            response = requests.get(API_URL)
            logging.info(response.status_code)
            if response.status_code == 200:
                logging.info(response.json()["response"]["meta"]["hits"])
                json_string = json.dumps(
                    {"results": response.json()["response"]["docs"]}
                )
                with open(
                    os.getenv("DATASETS_FOLDER_NY_TIMES_ORIGINAL")
                    + "the_ny_times_"
                    + str(current_year)
                    + "-"
                    + str(current_month)
                    + ".json",
                    "w",
                ) as outfile:
                    outfile.write(json_string)
            current_month += 1
        else:
            current_month = 1
            current_year += 1
    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    functions.logging_functions.create_log_file("get_dataset_ny_times")
    main()
