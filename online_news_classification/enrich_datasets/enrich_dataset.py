import glob
import logging
import os
import shutil
import time
from multiprocessing import Pool

import pandas as pd
from dotenv import load_dotenv
from send2trash import send2trash

from online_news_classification import constants
from online_news_classification.functions import enrich, manage_datasets, setup

load_dotenv()


def extract_integer(filename):
    if filename.endswith(".csv"):
        return int(filename.split(".")[0].split("_")[1])
    else:
        return -1


def get_key(fp):
    filename = os.path.splitext(os.path.basename(fp))[0]
    int_part = filename.split("_")[1]
    return int(int_part)


def enrich_dataset(filename):
    basename = os.path.splitext(os.path.basename(filename))[0]
    args = setup.get_arg_parser_enrich().parse_args()
    (
        start_time,
        refined,
        nlp,
        stop_words,
    ) = setup.initilize_with_models(f"enrich_{str(args.capitalization)}_{basename}")
    logging.info(filename)
    output_file = (
        f"{args.output_dir}enriched_{str(args.capitalization)}_"
        + f"{basename}{constants.FILE_EXTENSION}"
    )
    logging.info(output_file)
    dataset = manage_datasets.read_csv_dataset(filename=filename, separator=";")
    dataset["title_entities"] = pd.Series(dtype="object")
    dataset["abstract_entities"] = pd.Series(dtype="object")
    dataset["entities"] = pd.Series(dtype="object")
    dataset = enrich.enrich(
        dataset=dataset,
        option=args.capitalization,
        refined=refined,
        stop_words=stop_words,
    )
    dataset = dataset.drop(["Unnamed: 0"], axis=1)
    logging.info(
        os.path.join(
            os.getcwd(),
            os.getenv("DATASETS_FOLDER")
            + os.path.join(args.tmp_dir, os.path.basename(filename)),
        )
    )
    logging.info(os.path.join(args.output_dir, os.path.basename(filename)))
    try:
        manage_datasets.save_dataset(dataset, output_file)
        if args.enrich_mode == "folder":
            send2trash(
                os.path.join(
                    os.getcwd(),
                    os.getenv("DATASETS_FOLDER")
                    + os.path.join(args.tmp_dir, os.path.basename(filename)),
                )
            )
    except OSError:
        logging.info("Erro no guardar ficheiro!")
    logging.info("--- %s seconds ---" % (time.time() - start_time))


def main():
    args = setup.get_arg_parser_enrich().parse_args()
    if args.enrich_mode == "folder":
        in_directory = os.path.join(
            os.getcwd(), os.getenv("DATASETS_FOLDER") + args.input_dir
        )
        tmp_directory = os.path.join(
            os.getcwd(), os.getenv("DATASETS_FOLDER") + args.tmp_dir
        )
        if args.dataset_format == "file":
            files_copy = sorted(
                glob.glob(f"{in_directory}{constants.FILE_EXTENSION_SEARCH}"),
                key=get_key,
            )
            files = sorted(
                glob.glob(f"{tmp_directory}{constants.FILE_EXTENSION_SEARCH}"),
                key=get_key,
            )

        else:
            files_copy = sorted(
                glob.glob(f"{in_directory}{constants.FILE_EXTENSION_SEARCH}")
            )
            files = sorted(
                glob.glob(f"{tmp_directory}{constants.FILE_EXTENSION_SEARCH}")
            )

        for file in files_copy:
            shutil.copy2(file, tmp_directory)

        pool = Pool(processes=args.num_processes)
        pool.map(
            enrich_dataset,
            [os.path.join(args.tmp_dir, os.path.basename(file)) for file in files],
        )
        pool.close()
        pool.join()
    else:
        enrich_dataset(args.input_dir)


if __name__ == "__main__":
    main()
