import logging
import os

import online_news_classification_lib.functions as functions
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def convert(f, args, output_file):
    dataset = functions.manage_datasets_functions.read_json_dataset(filename=f)
    data = pd.json_normalize(dataset["results"][0])
    logging.info(len(data))
    data["title"] = data["headline.main"]
    data["category"] = data["section_name"]
    data = data[data["title"] != ""]
    data = data[data["abstract"] != ""]
    data = data[data["category"] != ""]
    data = data.drop(
        [
            "multimedia",
            "print_section",
            "snippet",
            "source",
            "lead_paragraph",
            "print_page",
            "word_count",
            "byline.person",
            "headline.kicker",
            "document_type",
            "news_desk",
            "headline.main",
            "headline.content_kicker",
            "headline.print_headline",
            "headline.name",
            "headline.seo",
            "type_of_material",
            "headline.sub",
            "byline.original",
            "byline.organization",
            "_id",
        ],
        axis=1,
    )
    data["final_tags"] = pd.Series(dtype="object")

    for index, row in data.iterrows():
        tags = []
        for tag in row["keywords"]:
            tags.append(str(tag["value"]))
        data.at[index, "final_tags"] = tags
        data.at[index, "final_tags"] = list(tags)

    data = data.drop(["keywords"], axis=1)

    functions.manage_datasets_functions.save_dataset(data, output_file + ".csv")


def main():
    args = functions.setup_functions.get_arg_parser_to_csv().parse_args()
    start_time = functions.setup_functions.initialize("ny_times_to_csv")
    if args.convert_mode == "folder":
        in_directory = os.path.join(
            os.getcwd(), os.getenv("DATASETS_FOLDER") + args.input
        )
        for filename in sorted(os.listdir(in_directory)):
            if filename.endswith(".json"):
                f = os.path.join(args.input, filename)
                output_file = os.path.join(args.output, os.path.splitext(filename)[0])
                convert(f, args, output_file)
    else:
        filename = os.path.basename(args.input)
        output_file = os.path.join(args.output, os.path.splitext(filename)[0])
        convert(args.input, args, output_file)

    functions.setup_functions.finalize(start_time)


if __name__ == "__main__":
    main()
