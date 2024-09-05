import json
import logging
import os
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

from online_news_classification.functions import setup

load_dotenv()


def main():
    args = setup.get_arg_parser_get_dataset_the_guardian().parse_args()
    start_time = setup.initialize("get_dataset")
    base_url = os.getenv("THE_GUARDIAN_BASE_API_URL")
    ORDER_BY = "newest"
    delta = timedelta(days=10)
    current_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    while current_date <= end_date:
        initial_date = current_date
        next_date = current_date + delta
        initial_date_str = str(initial_date.strftime("%Y-%m-%d"))
        next_date_str = str(next_date.strftime("%Y-%m-%d"))
        API_URL = (
            f"{base_url}&from-date={initial_date_str}&to-date={next_date_str}"
            + f"&order-by={ORDER_BY}&page-size=50&show-fields=trailText%2Cheadline"
            + "&show-tags=keyword"
        )
        response = requests.get(API_URL)
        logging.info(response.status_code)
        final_results = []
        if response.status_code == 200:
            number_of_pages = response.json()["response"]["pages"]
            logging.info(number_of_pages)
            for page in range(number_of_pages):
                url = (
                    f"{base_url}&from-date={initial_date_str}&to-date={next_date_str}"
                    + f"&order-by={ORDER_BY}&page-size=50&page={str(page)}&"
                    + "show-fields=trailText%2Cheadline&show-tags=keyword"
                )

                page_response = requests.get(url)
                logging.info(page_response)
                if page_response.status_code == 200:
                    results = page_response.json()["response"]["results"]
                    logging.info(results)
                    final_results = final_results + results
        if len(final_results) > 0:
            json_string = json.dumps({"results": final_results})
            exist = os.path.exists(
                os.getenv("DATASETS_FOLDER_THE_GUARDIAN_ORIGINAL_JSON")
            )
            if not exist:
                os.makedirs(os.getenv("DATASETS_FOLDER_THE_GUARDIAN_ORIGINAL_JSON"))
            with open(
                os.getenv("DATASETS_FOLDER_THE_GUARDIAN_ORIGINAL_JSON")
                + "/the_guardian_"
                + str(initial_date.strftime("%Y-%m-%d"))
                + str(next_date.strftime("%Y-%m-%d"))
                + ".json",
                "w",
            ) as outfile:
                outfile.write(json_string)
        current_date = next_date + timedelta(days=1)
    setup.finalize(start_time)


if __name__ == "__main__":
    main()
