import os

import pandas as pd
from dotenv import load_dotenv

from online_news_classification.functions import manage_datasets, setup
from bs4 import BeautifulSoup
import logging

load_dotenv()

def remove_html_tags(text):
    if isinstance(text, str):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()
    else:
        return text  # Return as is if not a string

def convert(f, output_file):
    dataset = manage_datasets.read_json_dataset(filename=f)
    data = pd.json_normalize(dataset["results"][0], max_level=1)
    data["title"] = data["fields.headline"]
    data["abstract"] = data["fields.trailText"]
    data['title'].fillna('')
    data['abstract'].fillna('')
    data["category"] = data["sectionId"]
    data['title'] = data['title'].replace(r'^\s*$', '', regex=True)
    data['abstract'] = data['abstract'].replace(r'^\s*$', '', regex=True)
    data = data.drop(
        [
            "id",
            "type",
            "webTitle",
            "fields.trailText",
            "fields.headline",
            "pillarName",
            "pillarId",
            "isHosted",
            "apiUrl",
            "webUrl",
        ],
        axis=1,
    )
    data["final_tags"] = pd.Series(dtype="object")
    # Convert the 'date' column to datetime format
    data['webPublicationDate'] = pd.to_datetime(data['webPublicationDate'])
    data['clean_title'] = data['title'].apply(remove_html_tags)
    data['clean_abstract'] = data['abstract'].apply(remove_html_tags)

    # Sort the DataFrame by the 'date' column in descending order
    df_sorted = data.sort_values(by='webPublicationDate', ascending=True)

    for index, row in df_sorted.iterrows():
        ids_array = [tag['id'] for tag in row['tags']]
        df_sorted.at[index, "final_tags"] = ids_array

    df_sorted = df_sorted.drop(["tags", "title", "abstract"], axis=1)
    df_sorted.rename(columns={'final_tags': 'tags'}, inplace=True)
    df_sorted.rename(columns={'clean_title': 'title'}, inplace=True)
    df_sorted.rename(columns={'clean_abstract': 'abstract'}, inplace=True)

    df_sorted['title'] = df_sorted['title'].str.strip()
    df_sorted['abstract'] = df_sorted['abstract'].str.strip()

    df_sorted['title'] = df_sorted['title'] .str.replace(';', ',')
    df_sorted['abstract'] = df_sorted['abstract'].str.replace(';', ',')

    manage_datasets.save_dataset(df_sorted, output_file + ".csv")


def main():
    args = setup.get_arg_parser_to_csv().parse_args()
    start_time = setup.initialize("the_guardian_to_csv")
    if args.convert_mode == "folder":
        in_directory = os.path.join(
            os.getcwd(), os.getenv("DATASETS_FOLDER") + args.input
        )
        for filename in sorted(os.listdir(in_directory)):
            logging.info(filename)
            if filename.endswith(".json"):
                f = os.path.join(args.input, filename)
                output_file = os.path.join(args.output, os.path.splitext(filename)[0])
                convert(f, output_file)
    else:
        filename = os.path.basename(args.input)
        output_file = os.path.join(args.output, os.path.splitext(filename)[0])
        convert(args.input, output_file)
    setup.finalize(start_time)


if __name__ == "__main__":
    main()
