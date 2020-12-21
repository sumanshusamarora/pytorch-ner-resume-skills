import pandas as pd
import numpy as np
import json
import os
import re
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchnlp.samplers import DistributedBatchSampler, BalancedSampler
from torchnlp.datasets.dataset import Dataset
import torchnlp
from torchnlp.encoders import LabelEncoder
from torchnlp.encoders.text import StaticTokenizerEncoder, CharacterEncoder
import nltk
import random
import mlflow
import mlflow.pytorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(fpath="../data/dataset_ready.pkl"):
    dataset_ready = pd.read_pickle(fpath)
    X_text_list = list(dataset_ready.X_text)
    y_binary_list = list(dataset_ready.y_binary)
    y_ner_list = list(dataset_ready.y_ner)
    return X_text_list, y_binary_list, y_ner_list


def get_POS_tags(X_text_list):
    X_tags = []
    for lst in X_text_list:
        postag = nltk.pos_tag([word if word.strip() != "" else "<OOS>" for word in lst])
        X_tags.append([tag[1] for tag in postag])
    return X_tags


def split_test_train(X_text_list, X_tags, y_binary_list, y_ner_list, split_size=0.3):
    test_index = random.choices(
        range(len(X_text_list)), k=int(split_size * len(X_text_list))
    )
    train_index = [ind for ind in range(len(X_text_list)) if ind not in test_index]

    X_text_list_train = [X_text_list[ind] for ind in train_index]
    X_text_list_test = [X_text_list[ind] for ind in test_index]

    X_tags_train = [X_tags[ind] for ind in train_index]
    X_tags_test = [X_tags[ind] for ind in test_index]

    y_binary_list_train = [y_binary_list[ind] for ind in train_index]
    y_binary_list_test = [y_binary_list[ind] for ind in test_index]

    y_ner_list_train = [y_ner_list[ind] for ind in train_index]
    y_ner_list_test = [y_ner_list[ind] for ind in test_index]

    return (
        X_text_list_train,
        X_text_list_test,
        X_tags_train,
        X_tags_test,
        y_binary_list_train,
        y_binary_list_test,
        y_ner_list_train,
        y_ner_list_test,
    )


def tokenize_sentence(X_text_list_train, X_text_list_test, MAX_SENTENCE_LEN):
    x_encoder = StaticTokenizerEncoder(
        sample=X_text_list_train, append_eos=False, tokenize=lambda x: x,
    )
    x_encoded_train = [x_encoder.encode(text) for text in X_text_list_train]
    x_padded_train = torch.LongTensor(
        pad_sequence(x_encoded_train, MAX_SENTENCE_LEN + 1)
    )

    x_encoded_test = [x_encoder.encode(text) for text in X_text_list_test]
    x_padded_test = torch.LongTensor(pad_sequence(x_encoded_test, MAX_SENTENCE_LEN + 1))

    if x_padded_train.shape[1] > x_padded_test.shape[1]:
        x_padded_test = torch.cat(
            (
                x_padded_test,
                torch.zeros(
                    x_padded_test.shape[0],
                    x_padded_train.shape[1] - x_padded_test.shape[1],
                ),
            ),
            dim=1,
        ).type(torch.long)

    return x_encoder, x_padded_train, x_padded_test


def tokenize_character(X_text_list_train, x_padded_train, x_padded_test, x_encoder):
    x_char_encoder = CharacterEncoder(sample=X_text_list_train, append_eos=False,)
    x_char_encoded_train = [
        [x_char_encoder.encode(x_encoder.index_to_token[word.item()]) for word in text]
        for text in x_padded_train
    ]
    MAX_WORD_LENGTH = max(
        [
            max([internal.shape[0] for internal in external])
            for external in x_char_encoded_train
        ]
    )
    # x_char_padded = max([max([internal.shape[0] for internal in external]) for external in x_char_encoded])
    # x_char_padded = torch.LongTensor(pad_sequence(x_char_encoded, MAX_SENTENCE_LEN+1))
    outer_list = []
    for lst in x_char_encoded_train:
        inner_list = []
        for ten in lst:
            res = torch.zeros(MAX_WORD_LENGTH, dtype=torch.long)
            res[: ten.shape[0]] = ten
            inner_list.append(res)
        outer_list.append(inner_list)

    x_char_padded_train = torch.stack([torch.stack(lst) for lst in outer_list])

    x_char_encoded_test = [
        [x_char_encoder.encode(x_encoder.index_to_token[word]) for word in text]
        for text in x_padded_test
    ]
    outer_list = []
    for lst in x_char_encoded_test:
        inner_list = []
        for ten in lst:
            res = torch.zeros(MAX_WORD_LENGTH, dtype=torch.long)
            res[: ten.shape[0]] = ten
            inner_list.append(res)
        outer_list.append(inner_list)

    x_char_padded_test = torch.stack([torch.stack(lst) for lst in outer_list])

    return x_char_encoder, x_char_padded_train, x_char_padded_test


def tokenize_pos_tags(X_tags_train, X_tags_test):
    x_postag_encoder = StaticTokenizerEncoder(
        sample=X_tags_train, append_eos=False, tokenize=lambda x: x,
    )
    x_postag_encoded_train = [x_postag_encoder.encode(text) for text in X_tags_train]
    x_postag_padded_train = torch.LongTensor(
        pad_sequence(x_postag_encoded_train, MAX_SENTENCE_LEN + 1)
    )
    # x_postag_ohe_train = torch.nn.functional.one_hot(x_postag_padded_train)

    x_postag_encoded_test = [x_postag_encoder.encode(text) for text in X_tags_test]
    x_postag_padded_test = torch.LongTensor(
        pad_sequence(x_postag_encoded_test, MAX_SENTENCE_LEN + 1)
    )

    if x_postag_padded_train.shape[1] > x_postag_padded_test.shape[1]:
        x_postag_padded_test = torch.cat(
            (
                x_postag_padded_test,
                torch.zeros(
                    x_postag_padded_test.shape[0],
                    x_postag_padded_train.shape[1] - x_postag_padded_test.shape[1],
                ),
            ),
            dim=1,
        ).type(torch.long)

    # x_postag_ohe_test = torch.nn.functional.one_hot(x_postag_padded_test)
    return x_postag_encoder, x_postag_padded_train, x_postag_padded_test


def encode_ner_y(y_ner_list_train, y_ner_list_test, CLASS_COUNT_DICT):
    y_ner_encoder = LabelEncoder(sample=CLASS_COUNT_DICT.keys())
    y_ner_encoded_train = [
        [y_ner_encoder.encode(label) for label in label_list]
        for label_list in y_ner_list_train
    ]
    y_ner_encoded_train = [torch.stack(tens) for tens in y_ner_encoded_train]
    y_ner_padded_train = torch.LongTensor(
        pad_sequence(y_ner_encoded_train, MAX_SENTENCE_LEN + 1)
    )

    y_ner_encoded_test = [
        [y_ner_encoder.encode(label) for label in label_list]
        for label_list in y_ner_list_test
    ]
    y_ner_encoded_test = [torch.stack(tens) for tens in y_ner_encoded_test]
    y_ner_padded_test = torch.LongTensor(
        pad_sequence(y_ner_encoded_test, MAX_SENTENCE_LEN + 1)
    )

    if y_ner_padded_train.shape[1] > y_ner_padded_test.shape[1]:
        y_ner_padded_test = torch.cat(
            (
                y_ner_padded_test,
                torch.zeros(
                    y_ner_padded_test.shape[0],
                    y_ner_padded_train.shape[1] - y_ner_padded_test.shape[1],
                ),
            ),
            dim=1,
        ).type(torch.long)

    return y_ner_padded_train, y_ner_padded_test


def encode_binary_y(y_binary_list_train, y_binary_list_test):
    y_binary_series_train = pd.Series(y_binary_list_train).astype("category")
    y_binary_encoded_train = torch.LongTensor(y_binary_series_train)

    y_binary_series_test = pd.Series(y_binary_list_test).astype("category")
    y_binary_encoded_test = torch.LongTensor(y_binary_series_test)

    y_binary_encode_map = {
        code: cat for code, cat in enumerate(y_binary_series_train.cat.categories)
    }
    return y_binary_encode_map, y_binary_encoded_train, y_binary_encoded_test


# Sample weights
def calculate_sample_weights(y_binary_encoded_train, y_ner_padded_train):
    ner_class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(torch.flatten(y_ner_padded_train).numpy()),
        y=torch.flatten(y_ner_padded_train).numpy(),
    )

    binary_class_weights = compute_class_weight(
        "balanced", classes=[0, 1], y=y_binary_encoded_train.numpy()
    )
    return binary_class_weights, ner_class_weights


# Model defintion
# Build Model
class EntityExtraction(nn.Module):
    def __init__(
        self,
        num_classes,
        word_embed_dim=124,
        tag_embed_dim=124,
        char_embed_dim=124,
        rnn_embed_dim=512,
        char_embedding=True,
        dropout_ratio=0.3,
    ):
        super().__init__()
        # self variables
        self.NUM_CLASSES = num_classes
        self.word_embed_dim = word_embed_dim
        self.tag_embed_dim = tag_embed_dim
        self.char_embed_dim = char_embed_dim
        self.rnn_embed_dim = rnn_embed_dim
        self.dropout_ratio = dropout_ratio

        # Embedding Layers
        self.word_embed = nn.Embedding(
            num_embeddings=x_encoder.vocab_size, embedding_dim=self.word_embed_dim
        )
        self.word_embed_drop = nn.Dropout(self.dropout_ratio)

        self.char_embed = nn.Embedding(
            num_embeddings=x_char_encoder.vocab_size, embedding_dim=self.char_embed_dim
        )
        self.char_embed_drop = nn.Dropout(self.dropout_ratio)

        self.postag_embed = nn.Embedding(
            num_embeddings=x_postag_encoder.vocab_size, embedding_dim=self.tag_embed_dim
        )
        self.tag_embed_drop = nn.Dropout(self.dropout_ratio)

        # CNN for character input
        self.conv_char = nn.Conv1d(
            in_channels=self.char_embed_dim, out_channels=52, kernel_size=3, padding=1
        )
        # self.maxpool_char = nn.MaxPool1d(kernel_size=3)

        # LSTM for concatenated input
        self.lstm_binary = nn.LSTM(
            input_size=5760,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        self.lstm_binary_drop = nn.Dropout(self.dropout_ratio)

        self.lstm_ner = nn.LSTM(
            input_size=5761,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        self.lstm_ner_drop = nn.Dropout(self.dropout_ratio)

        # Linear layer
        self.linear_binary = nn.Linear(in_features=1024, out_features=1)

        # self.linear1 = nn.Linear(in_features=1024, out_features=512)
        self.linear_ner = nn.Linear(
            in_features=1024, out_features=self.NUM_CLASSES + 1
        )  # +1 for padding 0

    def forward(self, x_word, x_char, x_pos):
        x_char_shape = x_char.shape
        batch_size = x_char_shape[0]

        word_out = self.word_embed(x_word)
        word_out = self.word_embed_drop(word_out)

        char_out = self.char_embed(x_char)
        char_out = self.char_embed_drop(char_out)

        tag_out = self.postag_embed(x_pos)
        tag_out = self.tag_embed_drop(tag_out)

        char_out_shape = char_out.shape
        char_out = char_out.view(
            char_out_shape[0], char_out_shape[1] * char_out_shape[2], char_out_shape[3]
        )
        char_out = self.conv_char(char_out.permute(0, 2, 1))
        char_out = char_out.view(char_out_shape[0], char_out_shape[1], -1)

        concat = torch.cat((word_out, char_out, tag_out), dim=2)
        concat = F.relu(concat)

        lstm_out, (h, c) = self.lstm_binary(concat)
        lstm_out = self.lstm_binary_drop(lstm_out)

        # Binary model result
        binary_out = self.linear_binary(lstm_out[:, -1, :])
        binary_out = torch.sigmoid(binary_out)
        binary_out_repeat = binary_out.repeat_interleave(concat.shape[1], 0).view(
            batch_size, concat.shape[1], -1
        )

        # Concatenate binary result with embeddings
        concat = torch.cat((concat, binary_out_repeat), dim=2)

        # NER LSTM
        ner_lstm_out, _ = self.lstm_ner(concat)
        ner_lstm_out = self.lstm_ner_drop(ner_lstm_out)

        ner_out = F.softmax(self.linear_ner(ner_lstm_out))

        return binary_out, ner_out


class ClassificationModelUtils:
    def __init__(
        self,
        dataloader_train,
        dataloader_test,
        binary_class_weights,
        ner_class_weights,
        cuda=True,
    ):
        if cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu")

        self.model = EntityExtraction(num_classes=NUM_CLASSES)
        self.model = self.model.to(self.device)

        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test

        self.binary_class_weights = binary_class_weights
        self.ner_class_weights = ner_class_weights

        self.criterion_crossentropy = nn.NLLLoss(
            weight=torch.FloatTensor(self.ner_class_weights).to(device)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

        self.epoch_losses = []
        self.epoch_accuracy = []
        self.epoch_recall = []
        self.epoch_precision = []
        self.epoch_f1s = []

        self.test_epoch_loss = []
        self.test_epoch_accuracy = []
        self.test_epoch_recall = []
        self.test_epoch_precision = []
        self.test_epoch_f1s = []

    def evaluate_classification_metrics(self, truth, prediction):
        precision = precision_score(truth, prediction)
        accuracy = accuracy_score(truth, prediction)
        f1 = f1_score(truth, prediction)
        recall = recall_score(truth, prediction)
        return accuracy, precision, recall, f1

    def plot_graphs(self, figsize=(16, 10)):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(2, 3, 1)
        ax.plot(self.epoch_losses, color="b", label="Train")
        ax.plot(self.test_epoch_loss, color="g", label="Test")
        ax.legend()
        ax.set_title("Losses")

        ax = fig.add_subplot(2, 3, 2)
        ax.plot(self.epoch_accuracy, color="b", label="Train")
        ax.plot(self.test_epoch_accuracy, color="g", label="Test")
        ax.legend()
        ax.set_title("Accuracy")

        ax = fig.add_subplot(2, 3, 3)
        ax.plot(self.epoch_precision, color="b", label="Train")
        ax.plot(self.test_epoch_precision, color="g", label="Test")
        ax.legend()
        ax.set_title("Precision")

        ax = fig.add_subplot(2, 3, 4)
        ax.plot(self.epoch_recall, color="b", label="Train")
        ax.plot(self.test_epoch_recall, color="g", label="Test")
        ax.legend()
        ax.set_title("Recall")

        ax = fig.add_subplot(2, 3, 6)
        ax.plot(self.epoch_f1s, color="b", label="Train")
        ax.plot(self.test_epoch_f1s, color="g", label="Test")
        ax.legend()
        ax.set_title("F1")

        plt.show()

    def validate(self):
        test_losses = []
        test_accs = []
        test_precisions = []
        test_recalls = []
        test_f1s = []
        test_binary_batch_all = None
        test_binary_out_result_all = None

        print("************Evaluating validation data now***************")
        for k, data_test in enumerate(self.dataloader_test):
            with torch.no_grad():
                test_weight_this_batch = (
                    torch.FloatTensor(
                        [
                            self.binary_class_weights[val]
                            for val in data_test["y_binary_encoded"].numpy()
                        ]
                    )
                    .to(self.device)
                    .unsqueeze(1)
                )
                test_criterion_binary = nn.BCELoss(weight=test_weight_this_batch)

                data_test["x_padded"] = data_test["x_padded"].to(self.device)
                data_test["x_char_padded"] = data_test["x_char_padded"].to(self.device)
                data_test["x_postag_padded"] = data_test["x_postag_padded"].to(
                    self.device
                )
                data_test["y_binary_encoded"] = (
                    data_test["y_binary_encoded"]
                    .type(torch.float)
                    .unsqueeze(1)
                    .to(self.device)
                )
                data_test["y_ner_padded"] = data_test["y_ner_padded"].to(self.device)

                test_binary_out, test_ner_out = self.model(
                    data_test["x_padded"],
                    data_test["x_char_padded"],
                    data_test["x_postag_padded"],
                )

                test_binary_loss = test_criterion_binary(
                    test_binary_out, data_test["y_binary_encoded"]
                )
                test_ner_loss = self.criterion_crossentropy(
                    test_ner_out.transpose(2, 1), data_test["y_ner_padded"]
                )

                # Loss
                test_loss = torch.mean(torch.stack((test_binary_loss, test_ner_loss)))
                test_losses.append(test_loss.item())

                # Evaluation Metrics
                test_binary_out_result = (
                    (test_binary_out >= 0.5).type(torch.int).squeeze(-1)
                )
                test_binary_out_result = np.array(
                    [y_binary_encode_map[v.item()] for v in test_binary_out_result]
                )
                test_binary_batch = (
                    data_test["y_binary_encoded"].to("cpu").type(torch.int).numpy()
                )

                if not test_binary_batch_all is None:
                    test_binary_batch_all = np.append(
                        test_binary_batch_all, test_binary_batch
                    )
                    test_binary_out_result_all = np.append(
                        test_binary_out_result_all, test_binary_out_result
                    )
                else:
                    test_binary_batch_all = test_binary_batch.copy()
                    test_binary_out_result_all = test_binary_out_result.copy()

                (
                    test_accuracy,
                    test_precision,
                    test_recall,
                    test_f1,
                ) = self.evaluate_classification_metrics(
                    test_binary_batch_all, test_binary_out_result_all
                )

                test_accs.append(test_accuracy)
                test_precisions.append(test_precision)
                test_recalls.append(test_f1)
                test_f1s.append(test_recall)

        self.test_epoch_loss.append(np.array(test_losses).mean())
        self.test_epoch_accuracy.append(test_accuracy)
        self.test_epoch_recall.append(test_recall)
        self.test_epoch_precision.append(test_precision)
        self.test_epoch_f1s.append(test_f1)

    def train(self, num_epochs=10):
        index_metric_append = int(len(dataloader_train) / 4)
        for epoch in range(num_epochs):
            print(
                f"\n\n------------------------- Epoch - {epoch + 1} -------------------------"
            )
            batch_losses = []
            batch_accuracy = []
            batch_f1s = []
            batch_recalls = []
            batch_precisions = []
            binary_batch_all = None
            binary_out_result_all = None

            for batch_num, data in enumerate(dataloader_train):

                weight_this_batch = (
                    torch.FloatTensor(
                        [
                            self.binary_class_weights[val]
                            for val in data["y_binary_encoded"].numpy()
                        ]
                    )
                    .to(self.device)
                    .unsqueeze(1)
                )
                criterion_binary = nn.BCELoss(weight=weight_this_batch)

                self.optimizer.zero_grad()

                data["x_padded"] = data["x_padded"].to(self.device)
                data["x_char_padded"] = data["x_char_padded"].to(self.device)
                data["x_postag_padded"] = data["x_postag_padded"].to(self.device)
                data["y_binary_encoded"] = (
                    data["y_binary_encoded"]
                    .type(torch.float)
                    .unsqueeze(1)
                    .to(self.device)
                )
                data["y_ner_padded"] = data["y_ner_padded"].to(self.device)

                binary_out, ner_out = self.model(
                    data["x_padded"], data["x_char_padded"], data["x_postag_padded"]
                )

                binary_loss = criterion_binary(binary_out, data["y_binary_encoded"])
                ner_loss = self.criterion_crossentropy(
                    ner_out.transpose(2, 1), data["y_ner_padded"]
                )

                # Loss
                loss = torch.mean(torch.stack((binary_loss, ner_loss)))
                batch_losses.append(loss.item())

                # Evaluation Metrics
                binary_out_result = (binary_out >= 0.5).type(torch.int).squeeze(-1)
                binary_out_result = np.array(
                    [y_binary_encode_map[v.item()] for v in binary_out_result]
                )
                binary_batch = (
                    data["y_binary_encoded"].to("cpu").type(torch.int).numpy()
                )

                if not binary_batch_all is None:
                    binary_batch_all = np.append(binary_batch_all, binary_batch)
                    binary_out_result_all = np.append(
                        binary_out_result_all, binary_out_result
                    )
                else:
                    binary_batch_all = binary_batch.copy()
                    binary_out_result_all = binary_out_result.copy()

                accuracy, precision, recall, f1 = self.evaluate_classification_metrics(
                    binary_batch_all, binary_out_result_all
                )

                batch_f1s.append(f1)
                batch_accuracy.append(accuracy)
                batch_recalls.append(recall)
                batch_precisions.append(precision)

                if batch_num % index_metric_append == 0 and batch_num != 0:
                    print(
                        f"--> Batch - {batch_num + 1}, "
                        + f"Loss - {np.array(batch_losses).mean():.4f} (B-{binary_loss.item():.4f}, N-{ner_loss.item():.4f}), "
                        + f"Accuracy - {accuracy:.2f}, "
                        + f"Recall - {recall:.2f}, "
                        + f"Precision - {precision:.2f}, "
                        + f"F1 - {f1:.2f}"
                    )

                loss.backward()
                self.optimizer.step()

            self.epoch_losses.append(np.array(batch_losses).mean())
            self.epoch_accuracy.append(accuracy)
            self.epoch_recall.append(recall)
            self.epoch_precision.append(precision)
            self.epoch_f1s.append(f1)

            self.validate()

            self.plot_graphs()


if __name__ == "__main__":
    EPOCHS = 20
    mlflow.set_experiment("PytorchDualLoss")
    with mlflow.start_run() as run:
        mlflow.log_param("EPOCHS", EPOCHS)
        # Load Data
        X_text_list, y_binary_list, y_ner_list = load_data("data/dataset_ready.pkl")

        # Get POS tags
        X_tags = get_POS_tags(X_text_list)

        # Split data in test and train plus return segregate as input lists
        (
            X_text_list_train,
            X_text_list_test,
            X_tags_train,
            X_tags_test,
            y_binary_list_train,
            y_binary_list_test,
            y_ner_list_train,
            y_ner_list_test,
        ) = split_test_train(
            X_text_list, X_tags, y_binary_list, y_ner_list, split_size=0.3
        )

        # Set some important parameters values
        MAX_SENTENCE_LEN = max([len(sentence) for sentence in X_text_list_train])
        ALL_LABELS = []
        _ = [[ALL_LABELS.append(label) for label in lst] for lst in y_ner_list_train]
        CLASS_COUNT_OUT = np.unique(ALL_LABELS, return_counts=True)
        CLASS_COUNT_DICT = dict(zip(CLASS_COUNT_OUT[0], CLASS_COUNT_OUT[1]))
        NUM_CLASSES = len([clas for clas in CLASS_COUNT_DICT.keys()])
        print(
            f"Max sentence length - {MAX_SENTENCE_LEN}, Total Classes = {NUM_CLASSES}"
        )

        mlflow.log_param("MAX_SENTENCE_LEN", MAX_SENTENCE_LEN)
        mlflow.log_param("NUM_CLASSES", NUM_CLASSES)

        # Tokenize Sentences
        x_encoder, x_padded_train, x_padded_test = tokenize_sentence(
            X_text_list_train, X_text_list_test, MAX_SENTENCE_LEN
        )

        # Tokenize Characters
        x_char_encoder, x_char_padded_train, x_char_padded_test = tokenize_character(
            X_text_list_train, x_padded_train, x_padded_test, x_encoder
        )

        # Tokenize Pos tags
        (
            x_postag_encoder,
            x_postag_padded_train,
            x_postag_padded_test,
        ) = tokenize_pos_tags(X_tags_train, X_tags_test)

        # Encode y NER
        y_ner_padded_train, y_ner_padded_test = encode_ner_y(
            y_ner_list_train, y_ner_list_test, CLASS_COUNT_DICT
        )

        # Encode y binary
        (
            y_binary_encode_map,
            y_binary_encoded_train,
            y_binary_encoded_test,
        ) = encode_binary_y(y_binary_list_train, y_binary_list_test)

        # Create train dataloader
        dataset_train = Dataset(
            [
                {
                    "x_padded": x_padded_train[i],
                    "x_char_padded": x_char_padded_train[i],
                    "x_postag_padded": x_postag_padded_train[i],
                    "y_ner_padded": y_ner_padded_train[i],
                    "y_binary_encoded": y_binary_encoded_train[i],
                }
                for i in range(x_padded_train.shape[0])
            ]
        )

        dataloader_train = DataLoader(
            dataset=dataset_train, batch_size=512, shuffle=True
        )

        # Create test dataloader
        dataset_test = Dataset(
            [
                {
                    "x_padded": x_padded_test[i],
                    "x_char_padded": x_char_padded_test[i],
                    "x_postag_padded": x_postag_padded_test[i],
                    "y_ner_padded": y_ner_padded_test[i],
                    "y_binary_encoded": y_binary_encoded_test[i],
                }
                for i in range(x_padded_test.shape[0])
            ]
        )

        dataloader_test = DataLoader(
            dataset=dataset_test, batch_size=512, shuffle=False
        )

        # Get sample weights
        binary_class_weights, ner_class_weights = calculate_sample_weights(
            y_binary_encoded_train, y_ner_padded_train
        )

        # Build model
        model_utils = ClassificationModelUtils(
            dataloader_train, dataloader_test, binary_class_weights, ner_class_weights
        )
        model_utils.train(EPOCHS)
        mlflow.log_param("Type", "WORD-CHAR-POS-CNN-RNN-BIN-NER")
        mlflow.pytorch.log_model(model_utils.model, "models")
        mlflow.log_metric("TestAccuracy", model_utils.test_epoch_accuracy[-1])
        mlflow.log_metric("TrainAccuracy", model_utils.epoch_accuracy[-1])

        mlflow.log_metric("TestPrecision", model_utils.test_epoch_precision[-1])
        mlflow.log_metric("TrainPrecision", model_utils.epoch_precision[-1])

        mlflow.log_metric("TestRecall", model_utils.test_epoch_recall[-1])
        mlflow.log_metric("TrainRecall", model_utils.epoch_recall[-1])

        mlflow.log_metric("TestF1", model_utils.test_epoch_f1s[-1])
        mlflow.log_metric("TrainF1", model_utils.epoch_f1s[-1])
