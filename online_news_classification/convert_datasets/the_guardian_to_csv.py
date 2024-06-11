import os

import pandas as pd
from dotenv import load_dotenv

from online_news_classification.lib.functions import manage_datasets, setup

load_dotenv()


def convert(f, args, output_file):
    dataset = manage_datasets.read_json_dataset(filename=f)
    data = pd.json_normalize(dataset["results"][0], max_level=1)
    data["title"] = data["fields.headline"]
    data["abstract"] = data["fields.trailText"]
    data["category"] = data["sectionId"]
    data = data[data["title"] != ""]
    data = data[data["abstract"] != ""]
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

    for index, row in data.iterrows():
        tags = []
        for tag in row["tags"]:
            tags.append(str(tag["id"]))
        data.at[index, "final_tags"] = tags
        data.at[index, "final_tags"] = list(tags)

    data = data.drop(["tags"], axis=1)

    manage_datasets.save_dataset(data, output_file + ".csv")


def main():
    args = setup.get_arg_parser_to_csv().parse_args()
    start_time = setup.initialize("the_guardian_to_csv")
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
    setup.finalize(start_time)


if __name__ == "__main__":
    main()
