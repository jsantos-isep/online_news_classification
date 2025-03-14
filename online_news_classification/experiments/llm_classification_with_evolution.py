import os
import dask.dataframe as dd
import torch
import logging
import torch.nn.functional as F
import time
import pandas as pd
import numpy as np
import dask
from dask import delayed
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch import autocast, nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import login
from online_news_classification.functions import setup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# CONFIGURATION & INITIAL SETUP
# ------------------------------
def setup_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_dotenv()
login(os.getenv("HUGGING_FACE_TOKEN"))
device = setup_device()
LOW_CONFIDENCE_THRESHOLD = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.85
window_size = 8
embed_model = SentenceTransformer("BAAI/bge-small-en").to(device)
cached_embeddings_by_category = {}


# ------------------------------
# MODEL UTILITIES
# ------------------------------
# Freeze all layers except the classification head
def freeze_all_layers_but_classifier(model):
    # Freeze all layers except the classifier
    for name, param in model.named_parameters():
        param.requires_grad = 'classifier' in name


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def classify_partition(df_partition, model, tokenizer, label_encoder):
    """Classifies a partition of data and tracks confidence/drift."""
    inputs = tokenizer(df_partition['tokenized_text'].tolist(), return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():  # Disable gradients for faster inference
        model.to(device)
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)

    df_partition["confidence"] = probs.max(dim=-1).values.cpu().numpy()
    df_partition["predicted_label_num"] = probs.argmax(dim=-1).cpu().numpy()
    df_partition["predicted_label"] = label_encoder.inverse_transform(probs.argmax(dim=-1).cpu().numpy())

    return df_partition


# # ------------------------------
# # CONFIDENCE CHECK
# # ------------------------------
# def check_confidence(texts, true_labels, true_tags, confidences, low_threshold=0.3, higher_threshold=0.75):
#     """Determine if the prediction confidence is below the threshold."""
#     mask_low = confidences < low_threshold
#     mask_high = confidences >= higher_threshold
#     uncertain_docs = [(t, l) for t, l, m in zip(texts, true_labels, mask_low) if m]
#     certain_docs = [(t, l, eval(tag)) for t, l, tag, m in zip(texts, true_labels, true_tags, mask_high) if m]
#     return np.sum(mask_low), uncertain_docs, np.sum(mask_high), certain_docs


# ------------------------------
# RETRAINING LOGIC
# ------------------------------
def retrain_model(model, tokenizer, candidate_labels, label_encoder, df_partition, category_column, evolve, new_categories, learning_rate=5e-5):
    """Retrain the main model with new specific labels."""
    logging.info(f"Retraining the model with: {len(df_partition)} documents")

    if df_partition.empty:
        return candidate_labels, model, tokenizer, label_encoder, df_partition

    new_candidate_labels = list(set(candidate_labels + df_partition[category_column].tolist()))

    # If there are changes in the number of candidate labels
    if len(new_candidate_labels) > len(candidate_labels):
        candidate_labels = new_candidate_labels
        label_encoder.fit(candidate_labels)
        # Replace the classifier layer
        model.num_labels = len(new_candidate_labels)
        model.config.num_labels = len(new_candidate_labels)
        model.classifier = torch.nn.Linear(model.config.hidden_size, len(new_candidate_labels)).to(device)

        # Reinitialize weights for the classifier
        torch.nn.init.xavier_uniform_(model.classifier.weight)

    ddf_text = df_partition['tokenized_text'].tolist()
    ddf_category = label_encoder.transform(df_partition[category_column]).tolist()
    dataset = TextDataset(ddf_text, ddf_category, tokenizer, 128)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    logging.info(f"Model num_labels: {model.config.num_labels}")

    model.train()

    # Training Loop
    for batch in dataloader:

        # Move input batch to GPU
        batch = {key: val.to(device) for key, val in batch.items()}

        # Pass the batch to the model
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    model.eval()

    return candidate_labels, model, tokenizer, label_encoder, df_partition


# ------------------------------
# LLM LABELLING FUNCTION
# ------------------------------
def llm_labelling(df_partition, generator, candidate_tags_by_labels, all_predictions_llm, all_true_labels_llm):
    logging.info(f"Starting labelling with {len(df_partition)} documents")
    for i, doc in df_partition.iterrows():
        prompt = f"""
        Given the following text and category, suggest a more specific subcategory. If a subcategory is not needed, return the original category.

        Text: {doc['tokenized_text']}
        Category: {doc['category']}

        Return the result in the following format:
        Suggested Subcategory: [subcategory or category]
        """
        with autocast("cuda"):
            response = generator(prompt, return_full_text=False, max_new_tokens=25, num_return_sequences=1)
            generated_text = response[0]['generated_text'].strip()
            if "Suggested Subcategory:" in generated_text:
                chosen_label = generated_text.split("Suggested Subcategory:")[1].strip()
                #chosen_label = next((label for label in possible_labels if label in generated_text), None)
                logging.info(f"Chosen label: {chosen_label}")
                if chosen_label and chosen_label != None:
                    if '.' not in doc['category'] and doc['category'] not in candidate_tags_by_labels:
                        candidate_tags_by_labels[doc['category']] = []
                    closest_label = get_closest_label(chosen_label, doc["category"], candidate_tags_by_labels)
                    logging.info(closest_label)
                    doc['true_category'] = doc['category']
                    logging.info(doc['category'])
                    logging.info(chosen_label)
                    all_true_labels_llm.append(doc['filtered_tags'])
                    all_predictions_llm.append(closest_label)
                    df_partition.at[i, 'category'] = closest_label
                    candidate_tags_by_labels[doc['category']].append(closest_label)

    return df_partition, candidate_tags_by_labels, all_predictions_llm, all_true_labels_llm

# Function to get cached embeddings or compute them if missing
def get_cached_embeddings(category, candidate_tags_by_labels):
    if category not in cached_embeddings_by_category:
        if category in candidate_tags_by_labels:
            possible_labels = candidate_tags_by_labels[category]
            cached_embeddings_by_category[category] = (
                possible_labels,  # Store labels
                embed_model.encode(possible_labels, convert_to_tensor=True, normalize_embeddings=True)
            # Store embeddings
            )
        else:
            return None  # Explicitly return None if category is not found in candidate_tags_by_labels

    return cached_embeddings_by_category.get(category)


# Function to find the closest label using embedding similarity
def get_closest_label(given_label, category, candidate_tags_by_labels):
    # Get possible labels and cached embeddings for the category
    cached_data = get_cached_embeddings(category, candidate_tags_by_labels)

    possible_labels, possible_embeddings = cached_data

    # Handle case where embeddings are not found
    if possible_embeddings.numel() == 0:
        return None  # Or return a default value, raise an error, etc.

    # Encode only the given label
    given_embedding = embed_model.encode(given_label, convert_to_tensor=True, normalize_embeddings=True)

    # Compute cosine similarity with cached embeddings
    cosine_similarities = torch.nn.functional.cosine_similarity(given_embedding, possible_embeddings)

    # Get the index of the highest similarity score
    closest_index = torch.argmax(cosine_similarities).item()

    # Return the closest label
    return possible_labels[closest_index]

# ------------------------------
# EVALUATION
# ------------------------------
def evaluate_model_performance(true_labels, predicted_labels):
    """Evaluate model performance using accuracy, precision, recall, F1-score and confusion matrix."""
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average="macro", zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average="macro", zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average="macro", zero_division=0)

    # Log the metrics
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-score: {f1:.4f}")

# ------------------------------
# MAIN PIPELINE
# ------------------------------
def main():
    args = setup.get_arg_parser_classification_active_learning().parse_args()
    start_time = setup.initialize(f"create_model_{args.experiment_type}_{args.category_level}_{args.experiment}_{str(args.model).split('/')[-1]}_{args.dataset}",
                                  )
    logging.info(f"Running into {device}...")

    logging.info(f"Experiment type: {args.experiment_type} || Category level: {args.category_level} || Experiment: {args.experiment} || Model: {args.model} || Dataset: {args.dataset}")

    low_confidence_or_drifted_docs = []

    # Load CSV file into Dask DataFrame

    candidate_tags_by_labels = {}
    if str(args.category_level) == "abstract":
        if str(args.experiment_type) == "baseline":
            ddf = dd.read_csv(
            f"../datasets/{args.dataset}/filtered_enriched_{args.dataset}_1.csv",
                sep=";",
                usecols=['tokenized_text', 'category'],
            )
            candidate_labels = ddf['category'].compute().unique().tolist()
            category_column = "category"
        elif str(args.experiment_type) == "evolution" or str(args.experiment_type == "evolution_deep_7"):
            ddf = dd.read_csv(
                f"../datasets/{args.dataset}/filtered_enriched_{args.dataset}_2.csv",
                sep=";",
                usecols=['tokenized_text', 'category'],
            )
    elif str(args.category_level) == "refined":
        if str(args.experiment_type) == "baseline":
            ddf = dd.read_csv(
                f"../datasets/{args.dataset}/filtered_enriched_{args.dataset}_1.csv",
                sep=";",
                usecols=['tokenized_text', 'filtered_tags'],
            )
            ddf = ddf.fillna("").astype(str)
            category_column = "filtered_tags"
            candidate_labels = ddf['filtered_tags'].compute().unique().tolist()
        elif str(args.experiment_type) == "evolution" or str(args.experiment_type == "evolution_deep_7"):
            ddf = dd.read_csv(
                f"../datasets/{args.dataset}/filtered_enriched_{args.dataset}_2.csv",
                sep=";",
                usecols=['tokenized_text', 'filtered_tags'],
            )
            ddf = ddf.fillna("").astype(str)
    elif str(args.category_level) == "abstract+refined":
        ddf = dd.read_csv(
            f"../datasets/{args.dataset}/filtered_enriched_{args.dataset}_2.csv",
            sep=";",
            usecols=['tokenized_text', 'category', 'filtered_tags'],
        )
        ddf = ddf.fillna("").astype(str)

    if str(args.experiment_type) == "baseline":
        model = AutoModelForSequenceClassification.from_pretrained(str(args.model), num_labels=len(candidate_labels),
                                                                   ignore_mismatched_sizes=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(str(args.model))

    elif str(args.experiment_type) == "evolution" or str(args.experiment_type == "evolution_deep_7"):
        # Load abstract model
        model_dir = f"../datasets/{args.dataset}/models/{str(args.model).split('/')[-1]}/abstract/{args.experiment}/baseline"


        # Load the fine-tuned model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        df = pd.read_csv(f'{model_dir}/candidate_labels.csv')

        candidate_labels = df['labels'].tolist()
        if str(args.category_level) == "abstract+refined":
            model_refined_dir = f"../datasets/{args.dataset}/models/{str(args.model).split('/')[-1]}/refined/{args.experiment}/baseline"

            df_refined = pd.read_csv(f'{model_refined_dir}/candidate_labels.csv')


            refined_candidate_labels = df_refined['labels'].tolist()
            logging.info(refined_candidate_labels)
            candidate_tags_by_labels = {key: [] for key in candidate_labels}


            for item in refined_candidate_labels:
                if str(item) != "":
                    key = str(item).split('.', 1)[0]  # Split only once at the first period
                    if key not in candidate_tags_by_labels.keys():
                        candidate_tags_by_labels[key] = []  # Initialize an empty list if the key is not in the dictionary
                    candidate_tags_by_labels[key].append(item)
        logging.info(candidate_tags_by_labels)
        logging.info(f"Candidate labels: {candidate_labels}")

    label_encoder = LabelEncoder()
    label_encoder.fit(candidate_labels)
    torch.manual_seed(0)

    # Freeze all layers except the classification head
    freeze_all_layers_but_classifier(model)

    logging.info(f"Dataset size: {len(ddf)}")
    rows_per_partition = 50
    if len(ddf) <= 20000:
        rows_per_partition = 1000
    elif 20000 < len(ddf) < 200000:
        rows_per_partition = 25

    npartitions = (len(ddf) // rows_per_partition) + int(len(ddf) % rows_per_partition > 0)
    logging.info(f"Number of partitions: {npartitions}")

    ddf = ddf.repartition(npartitions=npartitions)

    ddf = ddf.persist(scheduler="single-threaded")

    partitions = ddf.to_delayed()

    all_true_labels_main_model = []
    all_predictions_main_model = []

    all_true_sub_labels_main_model = []
    all_predictions_sub_main_model = []

    all_true_labels_llm = []
    all_predictions_llm = []

    high_conf_df = pd.DataFrame()

    size = len(partitions)
    midpoint = size // 2

    if str(args.experiment_type) != "baseline":
        generator = pipeline("text-generation", model=args.generative_llm,
                             trust_remote_code=True, device=device)

    for i, partition in enumerate(partitions):
        start_time_partition = time.time()
        df_partition = partition.compute()

        logging.info(f"Partition {i} of {size}")

        df_partition['chosen_label_by_llm'] = ''


        # Classify partition using model
        df_partition = classify_partition(df_partition, model, tokenizer, label_encoder)

        if str(args.category_level) == "abstract":
            all_true_labels_main_model.extend(df_partition["category"])
            all_predictions_main_model.extend(df_partition["predicted_label"])
        elif str(args.category_level) == "refined":
            all_true_sub_labels_main_model.extend(df_partition['filtered_tags'])
            all_predictions_sub_main_model.extend(df_partition['predicted_label'])
        elif str(args.category_level) == "abstract+refined":
            all_true_labels_main_model.extend(df_partition['category'])
            df_partition['first_part'] = df_partition['predicted_label'].str.split('.').str[0]
            all_predictions_main_model.extend(df_partition['first_part'])
            df_partition['second_part'] = df_partition['predicted_label'].str.split('.').str[1]
            all_true_sub_labels_main_model.extend(df_partition['filtered_tags'].fillna(''))
            all_predictions_sub_main_model.extend(df_partition['second_part'].fillna(''))

        # Different methods for retraining the model
        if str(args.experiment) == 'always_retrain':
            if str(args.experiment_type) == "evolution" or str(args.experiment_type == "evolution_deep_7"):
                category_column = "category"
                if args.category_level == "refined":
                    category_column = "filtered_tags"
                candidate_labels, model, tokenizer, label_encoder, df_partition = retrain_model(model, tokenizer,
                                                                                                candidate_labels,
                                                                                                label_encoder,
                                                                                                df_partition,
                                                                                                category_column,
                                                                                                False, [])

            else:
                candidate_labels, model, tokenizer, label_encoder, df_partition = retrain_model(model, tokenizer,
                                                                                           candidate_labels,
                                                                                           label_encoder, df_partition, category_column, False, [])

        elif str(args.experiment) == 'confidence_70':
            low_conf_df = df_partition[df_partition["confidence"] < 0.70]

            if len(low_conf_df) > 0:
                if str(args.experiment_type) == "evolution" or str(args.experiment_type == "evolution_deep_7"):
                    category_column = "category"
                    if args.category_level == "refined":
                        category_column = "filtered_tags"
                    candidate_labels, model, tokenizer, label_encoder, df_partition = retrain_model(model,
                                                                                                    tokenizer,
                                                                                                    candidate_labels,
                                                                                                    label_encoder,
                                                                                                    low_conf_df,
                                                                                                    category_column,
                                                                                                    False, [])
                else:
                    # Retrain the model if the confidence rate is higher than threshold
                    candidate_labels, model, tokenizer, label_encoder, low_conf_df = retrain_model(model, tokenizer,
                                                                                           candidate_labels,
                                                                                           label_encoder, low_conf_df, category_column, False, [])
        elif str(args.experiment) == 'confidence_80':
            low_conf_df = df_partition[df_partition["confidence"] < 0.80]

            if len(low_conf_df) > 0:
                if str(args.experiment_type) == "evolution" or str(args.experiment_type == "evolution_deep_7"):
                    candidate_labels, model, tokenizer, label_encoder, df_partition = retrain_model(model,
                                                                                                    tokenizer,
                                                                                                    candidate_labels,
                                                                                                    label_encoder,
                                                                                                    low_conf_df,
                                                                                                    "category",
                                                                                                    False, [])
                else:
                    # Retrain the model if the confidence rate is higher than threshold
                    candidate_labels, model, tokenizer, label_encoder, low_conf_df = retrain_model(model, tokenizer,
                                                                                           candidate_labels,
                                                                                           label_encoder, low_conf_df, category_column, False, [])

        # Improve the number of labels using the ontology
        if (str(args.experiment_type) == "evolution" or str(args.experiment_type == "evolution_deep_7")) and str(args.category_level) == 'abstract+refined':
            high_conf_df = df_partition[df_partition["confidence"] > 0.7]
            logging.info(f"High confidence count: {len(high_conf_df)}")
            if len(high_conf_df) > 0:
                high_conf_df, candidate_tags_by_labels, all_predictions_llm, all_true_labels_llm = llm_labelling(high_conf_df, generator,
                                                                       candidate_tags_by_labels, all_predictions_llm, all_true_labels_llm)
                logging.info(high_conf_df)
                candidate_labels, model, tokenizer, label_encoder, high_conf_df = retrain_model(model, tokenizer,
                                                                                                candidate_labels,
                                                                                                label_encoder,
                                                                                                high_conf_df, 'category', False, [])

                logging.info(f"Candidate labels: {candidate_labels}")
                high_conf_df = []

        # Evaluate the performance of simulated human classification
        logging.info("--- %s seconds ---" % (time.time() - start_time_partition))

    output_dir = f"../datasets/{args.dataset}/models/{str(args.model).split('/')[-1]}/{args.category_level}/{args.experiment}/{args.experiment_type}"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Model saved to {output_dir}")

    if str(args.experiment_type) == "baseline":
        df = pd.DataFrame(candidate_labels, columns=['labels'])
        df.replace("", np.nan, inplace=True)
        df.dropna(how='any', inplace=True)
        df = df.dropna()

        # Write the DataFrame to a CSV file
        df.to_csv(f"{output_dir}/candidate_labels.csv", index=False)

    if str(args.category_level) == 'abstract':
        evaluate_model_performance(all_true_labels_main_model, all_predictions_main_model)
    elif str(args.category_level) == 'refined':
        logging.info(f"Evaluation refined")
        evaluate_model_performance(all_true_sub_labels_main_model, all_predictions_sub_main_model)

    elif str(args.category_level) == 'abstract+refined':
        all_predictions_main_model = ['' if item is None else item for item in all_predictions_main_model]
        evaluate_model_performance(all_true_labels_main_model, all_predictions_main_model)
        logging.info(f"Evaluation refined")
        all_predictions_sub_main_model = ['' if item is None else item for item in all_predictions_sub_main_model]
        evaluate_model_performance(all_true_sub_labels_main_model, all_predictions_sub_main_model)
        logging.info(all_true_labels_llm)
        all_true_labels_llm = ['' if item is None else item for item in all_true_labels_llm]
        all_predictions_llm = ['' if item is None else item for item in all_predictions_llm]
        if len(all_predictions_llm) > 0:
            logging.info(f"Accuracy of LLM (human): {accuracy_score(all_true_labels_llm, all_predictions_llm)}")
        else:
            logging.info(f"Accuracy of LLM (human): 0")



    logging.info("--- %s seconds ---" % (time.time() - start_time))





if __name__ == "__main__":
    main()