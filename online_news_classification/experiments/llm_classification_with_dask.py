import ast
import logging
import os
import time

import dask.dataframe as dd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer

from online_news_classification.functions import prequential, setup

load_dotenv()


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length  # Define a maximum length for padding

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize the text with padding
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',  # Pad to max_length
            max_length=self.max_length,  # Specify maximum length
            return_tensors='pt'  # Return PyTorch tensors
        )
        # Extract the required tensors and return as a dictionary
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        # Remove the batch dimension
        # Convert label to tensor
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# Freeze all layers except the classification head
def freeze_all_layers_but_classifier(model):
    for param in model.base_model.parameters():
        param.requires_grad = False  # Freeze all parameters of the base model

    for param in model.classifier.parameters():
        # Only fine-tune the parameters of the classifier head
        param.requires_grad = True


# Function to predict labels and calculate metrics for each partition
def predict_and_evaluate(df_partition, text_type, tokenizer, model, label_encoder, 
                         all_predictions, all_true_categories, device, args, 
                         candidate_labels, all_class_accuracies, count_classes):
    start_time = time.time()
    texts = df_partition[text_type].tolist()
    true_labels = df_partition['category'].tolist()

    # Tokenize the input texts
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

    # Move the inputs to the GPU
    # Ensure all input tensors are on the same device
    inputs = {key: value.to(device) for key, value in inputs.items()}  

    # Perform forward pass to get logits (no gradient required for predictions)
    outputs = model(**inputs)
    logits = outputs.logits

    # Get predictios from logits
    predictions = torch.argmax(logits, dim=-1).tolist()

    if args.dataset != 'ag_news':
        predictions = label_encoder.inverse_transform(predictions)

    # Update overall predictions and true categories
    all_predictions.extend(predictions)
    all_true_categories.extend(true_labels)

    # Calculate metrics before training
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    precision = precision_score(true_labels, predictions, average='weighted',
                                zero_division=0)
    recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    class_report = classification_report(true_labels, predictions, output_dict=True,
                                         zero_division=0)

    result = pd.DataFrame({})
    if args.experiment == 'without_fine_tuning':
        result = pd.DataFrame({
            'accuracy': [accuracy],
            'f1_score': [f1],
            'precision': [precision],
            'recall': [recall],
            'classification_report': [class_report]
        })

    elif args.experiment == 'with_fine_tuning_all':
        model.train()  # Ensure model is in training mode during fine-tuning

        # Initialize optimizer and loss function
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=5e-5)  # Learning rate can be adjusted
        criterion = nn.CrossEntropyLoss().to(device)

        # Calculate the loss (for fine-tuning)
        encoded_true_labels = label_encoder.transform(true_labels)
        loss = criterion(logits, torch.tensor(encoded_true_labels).to(device))

        # Backpropagation and optimization
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Backpropagate to compute gradients
        optimizer.step()  # Update classifier's weights

        result = pd.DataFrame({
            'loss': [loss.item()],
            'accuracy': [accuracy],
            'f1_score': [f1],
            'precision': [precision],
            'recall': [recall],
            'classification_report': [class_report]
        })
        model.eval()

    elif args.experiment == 'with_historical_data':
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=candidate_labels)

        # Extract the diagonal elements (true positives for each class)
        true_positives = np.diag(cm)

        # Sum along rows to get the total actual instances for each class
        total_actuals = cm.sum(axis=1)

        class_accuracies = []
        for i in range(len(candidate_labels)):
            if total_actuals[i] == 0:
                class_accuracies.append(0)
                all_class_accuracies[i] += 0
            else:
                class_accuracies.append(true_positives[i]/total_actuals[i])
                all_class_accuracies[i] += true_positives[i]/total_actuals[i]

        df = pd.DataFrame(all_class_accuracies,
                          index=candidate_labels, columns=['value'])

        # Sort by values in ascending order
        df_sorted = df.sort_values(by='value')

        # Calculate the number of rows for the bottom 10%

        filtered_df = df_sorted[df_sorted['value'] < 
                                (int(os.getenv("ACCURACY_LIMIT"))/100)]

        percentage = int(os.getenv('PERCENTAGE_OF_CLASSES')) / 100
        num_bottom_10_percent = max(1, int(len(filtered_df) * percentage))

        # Select the bottom 10%
        bottom_10_percent_df = filtered_df.head(num_bottom_10_percent)

        max_length = 128  # Set a max length for the tokenizer
        for index in bottom_10_percent_df.index:
            category_counts = pd.Series(true_labels).value_counts()
            if index not in count_classes:
                count_classes[index] = 0

            if index in category_counts:
                count_classes[index] += category_counts[index]
            if count_classes[index] >= int(os.getenv("WINDOW_FOR_RETRAINING")):
                count_classes[index] = 0  # Reset counter for t
                logging.info(f"Training on class: {index}")
                ddf = dd.read_csv(args.input_file, sep=";", usecols=[text_type, 
                                                                     'category'])
                if os.getenv('DOCUMENTS_TO_RETRAIN') == 'oldest':
                    ddf_filtered = ddf[ddf['category'] == index].head(
                        int(os.getenv('NUMBER_OF_DOCUMENT_TO_RETRAIN')))
                elif os.getenv('DOCUMENTS_TO_RETRAIN') == 'newest':
                    ddf_filtered = ddf[ddf['category'] == index].tail(
                        int(os.getenv('NUMBER_OF_DOCUMENT_TO_RETRAIN')))

                if not ddf_filtered.empty:
                    model.to(device)
                    model.train()

                    # Prepare the data
                    ddf_filtered_text = ddf_filtered[text_type].tolist()
                    ddf_filtered_category = label_encoder.transform(
                        ddf_filtered['category']).tolist()

                    # Create the dataset
                    dataset = TextDataset(ddf_filtered_text, ddf_filtered_category, 
                                          tokenizer, max_length)
                    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

                    optimizer = AdamW(filter(lambda p: p.requires_grad, 
                                             model.parameters()), lr=5e-5)

                    for batch in train_dataloader:
                        optimizer.zero_grad()

                        # Move input batch to the GPU
                        batch = {key: val.to(device) for key, val in batch.items()}

                        outputs = model(**batch)  # Pass the batch to the model
                        loss = outputs.loss
                        loss.backward()
                        optimizer.step()

                    model.eval()

                    # evaluate the performance of the model on the same partition
                    true_labels = df_partition['category'].tolist()

                    # Tokenize the input texts
                    inputs = tokenizer(texts, return_tensors='pt', padding=True, 
                                       truncation=True)

                    # Move the inputs to the GPU
                    inputs = {key: value.to(device) for key, value in
                              inputs.items()}

                    # Perform forward pass to get logits
                    # (no gradient required for predictions)
                    outputs = model(**inputs)
                    logits = outputs.logits

                    # Get predictios from logits
                    predictions = torch.argmax(logits, dim=-1).tolist()

                    if args.dataset != 'ag_news':
                        predictions = label_encoder.inverse_transform(predictions)

    logging.info("--- %s seconds ---" % (time.time() - start_time))
    return result, model


def main():
    args = setup.get_arg_parser_llm_classification().parse_args()
    start_time = setup.initialize(
        f"{args.dataset}_{args.dataset_type}_{str(args.model).split('/')[-1]}_{args.experiment}",)

    # Initialize the prequential values
    soma, soma_a, nr_a, soma_w = 0, 0, 0, 0
    preq, preq_a, preq_w = [], [], []
    alpha, wind = 0.99, 500

    model_name = args.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running into {device}...")

    candidate_labels = ast.literal_eval(
        os.getenv(str(args.dataset).upper() + "_CATEGORIES")
    )
    label_encoder = LabelEncoder()

    text_type = 'enriched_text' if str(args.dataset_type) == 'enriched' else 'tokenized_text'

    # Load the pre-trained model and tokenizer
    model_name = args.model
    torch.manual_seed(0)
    if args.experiment == 'reuse_model_fine_tuned':
        # Specify the directory where the model is saved
        model_dir = f"{args.fine_tuned}/{str(args.model).split('/')[-1]}_{args.dataset}_fine_tuned"

        # Load the fine-tuned model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_dir,
                                                                   num_labels=len(candidate_labels)).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                   num_labels=len(candidate_labels), ignore_mismatched_sizes=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Freeze all layers except the classification head
    freeze_all_layers_but_classifier(model)

    # Load CSV file into Dask DataFrame
    ddf = dd.read_csv(args.input_file, sep=";", usecols=['tokenized_text',
                                                         'enriched_text', 'category'])

    # Partition the data into chunks (~1000 documents per partition)
    logging.info(len(ddf))
    rows_per_partition = 50
    if len(ddf) < 20000:
        rows_per_partition = 1000
    elif 20001 < len(ddf) < 200000:
        rows_per_partition = 25

    npartitions = ddf.shape[0].compute() // rows_per_partition + 1
    logging.info(npartitions)
    ddf = ddf.repartition(npartitions=npartitions)

    # Step 1: Persist the Dask DataFrame in memory
    ddf = ddf.persist(scheduler='single-threaded')

    # Step 2: Convert Dask DataFrame partitions into delayed objects (this will retain the order)
    partitions = ddf.to_delayed()

    all_predictions = []
    all_true_categories = []
    count_classes = {}
    all_class_accuracies = [0]*len(candidate_labels)
    number_of_documents = 0

    # Step 3: Sequentially process each partition
    for i, partition in enumerate(partitions):
        # Compute the partition to a Pandas DataFrame
        df_partition = partition.compute()
        number_of_documents += len(df_partition)

        # Apply your custom processing on the Pandas DataFrame
        result, model = predict_and_evaluate(df_partition, text_type=text_type,
                                             tokenizer=tokenizer, model=model,
                                             label_encoder=label_encoder,
                                             all_predictions=all_predictions,
                                             all_true_categories=all_true_categories,
                                             device=device, args=args,
                                             candidate_labels=candidate_labels,
                                             all_class_accuracies=all_class_accuracies,
                                             count_classes=count_classes)
        if i % 10 == 0 or i == len(partitions) - 1:
            result.to_csv(f"{args.classification_reports_folder}/{args.dataset}/" +
                          f"{str(args.model).split('/')[-1]}_{args.dataset_type}" +
                          f"_partition_{i + 1}.csv", index=False)

    if args.experiment != 'without_fine_tuning':
        # Calculate prequential
        all_preds = []
        for i in range(number_of_documents):
            val = 0 if all_predictions[i] == all_true_categories[i] else 1
            all_preds.append(val)
            preq_val, preq_a_val, preq_w_val, soma, soma_a, nr_a, soma_w = prequential.update_prequential_metrics(val, i, all_preds, soma, alpha, soma_a, nr_a, wind, soma_w)
            preq.append(preq_val)
            preq_a.append(preq_a_val)
            preq_w.append(preq_w_val)

        prequential.save_plot_and_results(args, preq, preq_a, preq_w, number_of_documents)

    if args.experiment == 'with_fine_tuning_all':
        # Save the model and tokenizer
        output_dir = f"{args.fine_tuned}/{str(args.model).split('/')[-1]}_{args.dataset}_fine_tuned"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logging.info(f"Model saved to {output_dir}")

    if args.experiment == 'with_historical_data':
        # Save the model and tokenizer
        output_dir = f"{args.fine_tuned}/{str(args.model).split('/')[-1]}_{args.dataset}_fine_tuned_with_historical_data"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logging.info(f"Model saved to {output_dir}")

    overall_accuracy = accuracy_score(all_true_categories, all_predictions)
    overall_f1 = f1_score(all_true_categories, all_predictions, average='weighted', zero_division=0)
    overall_precision = precision_score(all_true_categories, all_predictions, average='weighted', zero_division=0)
    overall_recall = recall_score(all_true_categories, all_predictions, average='weighted', zero_division=0)
    logging.info(f'Number of documents: {len(all_true_categories)}')
    logging.info(f'Overall Accuracy: {overall_accuracy:.4f}')
    logging.info(f'Overall F1: {overall_f1:.4f}')
    logging.info(f'Overall Precision: {overall_precision:.4f}')
    logging.info(f'Overall Recall: {overall_recall:.4f}')

    result = pd.DataFrame({
        'experiment': [args.experiment],
        'dataset': [args.dataset],
        'number of documents': [len(all_true_categories)],
        'overall accuracy': [f'{float(overall_accuracy)*100:.2f}'],
        'overall F1': [f'{float(overall_f1)*100:.2f}'],
        'overall precision': [f'{float(overall_precision)*100:.2f}'],
        'overall recall': [f'{float(overall_recall)*100:.2f}'],
        'execution time': [f'{time.time() - start_time:.2f}']

    })

    results_path = f"{os.getenv('RESULTS_FOLDER')}{args.dataset}/{str(args.model).split('/')[-1]}/"
    # Check if the folder exists
    if not os.path.exists(results_path):
        # Create the folder if it does not exist
        os.makedirs(results_path)

    result.to_excel(f"{results_path}/{args.dataset}_{str(args.model).split('/')[-1]}_{args.dataset_type}_{args.experiment}_results.xlsx", index=False)  # index=False to avoid writing row numbers

    logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
     main()
