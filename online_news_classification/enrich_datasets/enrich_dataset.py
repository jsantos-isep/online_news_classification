import glob
import logging
import os
import shutil
import time
from multiprocessing import Pool

import online_news_classification_lib.functions as functions
import pandas as pd
from dotenv import load_dotenv
from send2trash import send2trash

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


def enrich(filename):
    args = functions.setup_functions.get_arg_parser_enrich().parse_args()
    (
        start_time,
        refined,
        nlp,
        stop_words,
    ) = functions.setup_functions.initilize_with_models(
        "enrich_"
        + str(args.capitalization)
        + "_"
        + os.path.splitext(os.path.basename(filename))[0]
    )
    logging.info(filename)
    output_file = (
        args.output_dir
        + "enriched_"
        + str(args.capitalization)
        + "_"
        + os.path.splitext(os.path.basename(filename))[0]
        + ".csv"
    )
    logging.info(output_file)
    dataset = functions.manage_datasets_functions.read_csv_dataset(
        filename=filename, separator=";"
    )
    dataset["title_entities"] = pd.Series(dtype="object")
    dataset["abstract_entities"] = pd.Series(dtype="object")
    dataset["entities"] = pd.Series(dtype="object")
    dataset = functions.enrich_functions.enrich(
        dataset=dataset,
        option=args.capitalization,
        refined=refined,
        nlp=nlp,
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
        functions.manage_datasets_functions.save_dataset(dataset, output_file)
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
    args = functions.setup_functions.get_arg_parser_enrich().parse_args()
    if args.enrich_mode == "folder":
        in_directory = os.path.join(
            os.getcwd(), os.getenv("DATASETS_FOLDER") + args.input_dir
        )
        tmp_directory = os.path.join(
            os.getcwd(), os.getenv("DATASETS_FOLDER") + args.tmp_dir
        )
        if args.dataset_format == "file":
            files_copy = sorted(glob.glob(in_directory + "*.csv"), key=get_key)
            files = sorted(glob.glob(tmp_directory + "*.csv"), key=get_key)

        else:
            files_copy = sorted(glob.glob(in_directory + "*.csv"))
            files = sorted(glob.glob(tmp_directory + "*.csv"))

        for file in files_copy:
            shutil.copy2(file, tmp_directory)

        pool = Pool(processes=args.num_processes)
        pool.map(
            enrich,
            [os.path.join(args.tmp_dir, os.path.basename(file)) for file in files],
        )
        pool.close()
        pool.join()
    else:
        enrich(args.input_dir)


if __name__ == "__main__":
    main()
