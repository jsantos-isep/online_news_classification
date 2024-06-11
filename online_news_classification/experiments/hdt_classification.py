import ast
import csv
import glob
import logging
import os
import pickle
import shutil
import string
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from functions import manage_datasets, setup
from natsort import natsorted, realsorted
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from river import compose, drift
from river import feature_extraction as fx
from river import metrics, stream, tree
from send2trash import send2trash

load_dotenv()


def get_key(fp):
    filename = os.path.splitext(os.path.basename(fp))[0]
    int_part = filename.split("_")[3]
    return int(int_part)


def extract_integer(filename):
    if filename.endswith(".csv"):
        return int(filename.split(".")[0].split("_")[1])
    else:
        return -1


def classify(args, files, model_pkl_file):
    logging.info("Starting original experiment with: %s", str(args))
    page_hinkley = drift.binary.DDM()
    ps = PorterStemmer()
    preds = []
    soma = 0
    preq = []
    drifts = []

    alpha = 0.99
    soma_a = 0
    nr_a = 0
    preq_a = []

    wind = 500
    soma_w = 0
    preq_w = []

    metric = metrics.Accuracy()
    accuracies = []
    index = 0

    if args.feature_extraction == "BoW":
        en_stops = set(stopwords.words("english"))
        feature_extraction = fx.BagOfWords(
            lowercase=True, strip_accents=False, on="text", stop_words=en_stops
        )
    else:
        feature_extraction = fx.TFIDF(lowercase=True, strip_accents=False, on="text")

    if args.classification_type == "adaptive":
        classifier = tree.HoeffdingAdaptiveTreeClassifier(
            grace_period=float(os.getenv("GRACE_PERIOD")),
            delta=float(os.getenv("DELTA")),
            split_criterion=os.getenv("SPLIT_CRITERION"),
            max_depth=1000,
            bootstrap_sampling=False,
            drift_detector=drift.binary.DDM(),
            nominal_attributes=["category"],
            seed=0,
        )
    else:
        classifier = tree.HoeffdingTreeClassifier(
            grace_period=float(os.getenv("GRACE_PERIOD")),
            delta=float(os.getenv("DELTA")),
            split_criterion=os.getenv("SPLIT_CRITERION"),
            max_depth=1000,
            nominal_attributes=["category"],
        )

    pipeline_original = compose.Pipeline(
        ("feature_extraction", feature_extraction), ("classifier", classifier)
    )

    for file in files:
        start_time = time.time()
        dataset = manage_datasets.load_dataset_classify(file)
        dataset["title_stemmed"] = pd.Series(dtype="string")
        dataset["text"] = pd.Series(dtype="string")

        target = dataset["category"]
        docs = dataset.drop(["category"], axis=1)

        # Perform the online classification loop
        for xi, yi in stream.iter_pandas(docs, target):
            # Preprocess the current instance
            if args.text == "title":
                text_no_punct = str(xi["title"]).translate(
                    str.maketrans("", "", string.punctuation)
                )
            elif args.text == "abstract":
                text_no_punct = str(xi["abstract"]).translate(
                    str.maketrans("", "", string.punctuation)
                )
            else:
                text_no_punct = str(xi["title"]).translate(
                    str.maketrans("", "", string.punctuation)
                ) + str(xi["abstract"]).translate(
                    str.maketrans("", "", string.punctuation)
                )
            word_tokens = word_tokenize(text_no_punct)
            text = []
            for word in word_tokens:
                text.append(ps.stem(word))
            stemming = " ".join([sub for sub in text])
            logging.info("Index = %s", index)
            # logging.info("Title = %s", xi["title"])
            # logging.info("Stemming = %s", stemming)
            xi["title_stemmed"] = stemming

            if args.dataset_type == "original":
                xi["text"] = stemming
            elif args.dataset_type == "enriched":
                if args.text == "title":
                    xi["text"] = xi["title_entities"]
                elif args.text == "abstract":
                    xi["text"] = xi["abstract_entities"]
                else:
                    xi["text"] = xi["title_entities"] + xi["abstract_entities"]
            else:
                if args.text == "title":
                    xi["title_entities"] = ast.literal_eval(xi["title_entities"])
                    entities = " ".join([sub for sub in xi["title_entities"]])
                    xi["text"] = stemming + " " + entities
                elif args.text == "abstract":
                    xi["text"] = stemming + xi["abstract_entities"]
                else:
                    xi["text"] = (
                        stemming
                        + " "
                        + xi["title_entities"]
                        + " "
                        + xi["abstract_entities"]
                    )

            logging.info(xi["text"])
            pipeline_original["feature_extraction"].learn_one(xi)
            transformed_doc = pipeline_original["feature_extraction"].transform_one(xi)
            logging.info("Feature extraction result = %s", transformed_doc)

            # Make predictions and update the evaluation metric using the classifier
            y_pred = pipeline_original["classifier"].predict_one(transformed_doc)
            metric.update(yi, y_pred)
            accuracies.append(metric.get().real)
            logging.info("Accuracy = %s", metric.get().real)

            if y_pred == yi:
                val = 0
            else:
                val = 1
            preds.append(val)
            soma += val
            preq.append(soma / (index + 1))

            soma_a = val + alpha * soma_a
            nr_a = 1 + alpha * nr_a
            preq_a.append(soma_a / nr_a)

            soma_w += val
            if index >= wind:
                soma_w = soma_w - preds[index - wind]
                preq_w.append(soma_w / 500)
            else:
                preq_w.append(soma_w / (index + 1))

            if args.classification_type == "adaptive":
                detector = pipeline_original["classifier"].drift_detector
            else:
                detector = page_hinkley

            _ = detector.update(val)
            if detector.drift_detected:
                logging.info(
                    "Change detected at index %s, input value: %s, predict value %s",
                    index,
                    yi,
                    y_pred,
                )
                drifts.append({"index": index, "input": yi, "predict": y_pred})

            # Update the classifier with the preprocessed features and the true label
            pipeline_original["classifier"].learn_one(transformed_doc, yi)
            index += 1

        summary_file = os.path.join(
            os.getcwd(),
            os.getenv("DATASETS_FOLDER")
            + args.results_dir
            + "/summary/"
            + os.path.splitext(os.path.basename(file))[0],
        )
        plot_file = os.path.join(
            os.getcwd(),
            os.getenv("DATASETS_FOLDER")
            + args.results_dir
            + "/plot/"
            + os.path.splitext(os.path.basename(file))[0],
        )
        plot_aux_file = os.path.join(
            os.getcwd(),
            os.getenv("DATASETS_FOLDER")
            + args.results_dir
            + "/plot_aux/"
            + os.path.splitext(os.path.basename(file))[0],
        )
        tree_file = os.path.join(
            os.getcwd(),
            os.getenv("DATASETS_FOLDER")
            + args.results_dir
            + "/tree/"
            + os.path.splitext(os.path.basename(file))[0],
        )

        # create plot
        fig, ax = plt.subplots(figsize=(40, 20))
        ax.plot(range(index), preq, label="Prequential")
        ax.plot(range(index), preq_a, label="Prequential Alpha")
        ax.plot(range(index), preq_w, label="Prequential Window")
        for d in drifts:
            plt.axvline(x=d["index"], color="r")
        ax.legend()
        plt.savefig(
            plot_file
            + "_"
            + str(args.capitalization)
            + "_"
            + args.classification_type
            + "_"
            + args.feature_extraction
            + "_"
            + args.text
            + "_"
            + args.dataset_type
            + "_plot.png"
        )

        # create summary
        with open(
            summary_file
            + "_"
            + str(args.capitalization)
            + "_"
            + args.classification_type
            + "_"
            + args.feature_extraction
            + "_"
            + args.text
            + "_"
            + args.dataset_type
            + "_summary.csv",
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(
                [
                    "nº documents",
                    "categories",
                    "mean_accuracy",
                    "time",
                    "drifts",
                    "summary",
                ]
            )
            writer.writerow(
                [
                    index,
                    dataset["category"].nunique(),
                    metric.get().real,
                    (time.time() - start_time),
                    len(drifts),
                    pipeline_original["classifier"].summary,
                ]
            )
            f.close()

        # create plot aux
        df = pd.DataFrame({"preq": preq, "preq_a": preq_a, "preq_w": preq_w})
        df.to_csv(
            plot_aux_file
            + "_"
            + str(args.capitalization)
            + "_"
            + args.classification_type
            + "_"
            + args.feature_extraction
            + "_"
            + args.text
            + "_"
            + args.dataset_type
            + "_plot_aux.csv",
            index=False,
        )

        # with open(
        # plot_aux_file
        # + "_"
        # + str(args.capitalization)
        # + "_"
        # + args.classification_type
        # + "_"
        # + args.feature_extraction
        # + "_"
        # + args.text
        # + "_plot_aux.csv",
        # "w",
        # newline=""
        # ) as f:

        #     writer = csv.writer(f, delimiter=";")
        #     writer.writerow(["preq", "preq_a", "preq_w"])
        #     writer.writerow([preq, preq_a, preq_w])
        #     f.close()

        with open(
            plot_aux_file
            + "_"
            + str(args.capitalization)
            + "_"
            + args.classification_type
            + "_"
            + args.feature_extraction
            + "_"
            + args.text
            + "_"
            + args.dataset_type
            + "_accuracy_aux.csv",
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["accuracy"])
            writer.writerow([accuracies])
            f.close()

        # create tree
        with open(
            tree_file
            + "_"
            + str(args.capitalization)
            + "_"
            + args.classification_type
            + "_"
            + args.feature_extraction
            + "_"
            + args.text
            + "_tree.dot",
            "w",
        ) as f:
            f.write(str(pipeline_original["classifier"].draw()))

        with open(model_pkl_file, "wb") as model_file:
            model_to_file = {
                "model": pipeline_original,
                "preq": preq,
                "preq_a": preq_a,
                "preq_w": preq_w,
                "accuracies": accuracies,
            }
            pickle.dump(model_to_file, model_file)
        logging.info(model_to_file)
        send2trash(file)


def main():
    args = setup.get_arg_parser_hdt_classification().parse_args()
    start_time = setup.initialize(
        "new_experiment_" + str(args.capitalization) + "_" + args.dataset
    )
    in_directory = os.path.join(
        os.getcwd(), os.getenv("DATASETS_FOLDER") + args.input_dir
    )
    tmp_directory = os.path.join(
        os.getcwd(), os.getenv("DATASETS_FOLDER") + args.tmp_dir
    )
    if args.dataset_format == "file":
        files_copy = realsorted(glob.glob(in_directory + "*.csv"))
        # logging.info(files_copy)
        for file in files_copy:
            shutil.copy2(file, tmp_directory)
        files = realsorted(glob.glob(tmp_directory + "*.csv"))
    else:
        files_copy = natsorted(glob.glob(in_directory + "*.csv"))
        # logging.info(files_copy)
        for file in files_copy:
            shutil.copy2(file, tmp_directory)
        files = natsorted(glob.glob(tmp_directory + "*.csv"))

    model_pkl_file = os.path.join(
        os.getcwd(),
        os.getenv("MODELS_FOLDER")
        + "model_dt_"
        + str(args.dataset)
        + "_"
        + str(args.capitalization)
        + "_"
        + args.classification_type
        + "_"
        + args.feature_extraction
        + "_"
        + args.text
        + ".pkl",
    )
    classify(args, files, model_pkl_file)

    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
