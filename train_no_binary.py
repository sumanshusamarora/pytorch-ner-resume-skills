"""

"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder as sk_le, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchcrf import CRF
from torchnlp.datasets.dataset import Dataset
from torchnlp.encoders import LabelEncoder
from torchnlp.encoders.text import StaticTokenizerEncoder, CharacterEncoder
import nltk
import random
import mlflow
import mlflow.pytorch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(fpath="../data/dataset_ready.pkl"):
    dataset_ready = pd.read_pickle(fpath)
    X_text_list = list(dataset_ready.X_text)
    y_ner_list = list(dataset_ready.y_ner)
    return X_text_list, y_ner_list

def get_POS_tags(X_text_list):
    X_tags = []
    for lst in X_text_list:
        postag = nltk.pos_tag([word if word.strip() != "" else '<OOS>' for word in lst])
        X_tags.append([tag[1] for tag in postag])
    return X_tags

def split_test_train(X_text_list, X_tags, y_ner_list, split_size=0.3):
    test_index = random.choices(range(len(X_text_list)), k=int(split_size * len(X_text_list)))
    train_index = [ind for ind in range(len(X_text_list)) if ind not in test_index]

    X_text_list_train = [X_text_list[ind] for ind in train_index]
    X_text_list_test = [X_text_list[ind] for ind in test_index]

    X_tags_train = [X_tags[ind] for ind in train_index]
    X_tags_test = [X_tags[ind] for ind in test_index]

    y_ner_list_train = [y_ner_list[ind] for ind in train_index]
    y_ner_list_test = [y_ner_list[ind] for ind in test_index]

    return X_text_list_train, X_text_list_test, X_tags_train, X_tags_test, y_ner_list_train, y_ner_list_test


def tokenize_sentence(X_text_list_train, X_text_list_test, MAX_SENTENCE_LEN):
    x_encoder = StaticTokenizerEncoder(sample=X_text_list_train,
                                       append_eos=False,
                                       tokenize=lambda x: x,
                                       )
    x_encoded_train = [x_encoder.encode(text) for text in X_text_list_train]
    x_padded_train = torch.LongTensor(pad_sequence(x_encoded_train, MAX_SENTENCE_LEN + 1))

    x_encoded_test = [x_encoder.encode(text) for text in X_text_list_test]
    x_padded_test = torch.LongTensor(pad_sequence(x_encoded_test, MAX_SENTENCE_LEN + 1))

    if x_padded_train.shape[1] > x_padded_test.shape[1]:
        x_padded_test = torch.cat(
            (x_padded_test, torch.zeros(x_padded_test.shape[0], x_padded_train.shape[1] - x_padded_test.shape[1])),
            dim=1).type(torch.long)

    return x_encoder, x_padded_train, x_padded_test

def tokenize_character(X_text_list_train, x_padded_train, x_padded_test, x_encoder):
    x_char_encoder = CharacterEncoder(sample=X_text_list_train,
                                      append_eos=False,
                                      )
    x_char_encoded_train = [[x_char_encoder.encode(x_encoder.index_to_token[word.item()]) for word in text] for text in
                            x_padded_train]
    MAX_WORD_LENGTH = max([max([internal.shape[0] for internal in external]) for external in x_char_encoded_train])
    # x_char_padded = max([max([internal.shape[0] for internal in external]) for external in x_char_encoded])
    # x_char_padded = torch.LongTensor(pad_sequence(x_char_encoded, MAX_SENTENCE_LEN+1))
    outer_list = []
    for lst in x_char_encoded_train:
        inner_list = []
        for ten in lst:
            res = torch.zeros(MAX_WORD_LENGTH, dtype=torch.long)
            res[:ten.shape[0]] = ten
            inner_list.append(res)
        outer_list.append(inner_list)

    x_char_padded_train = torch.stack([torch.stack(lst) for lst in outer_list])

    x_char_encoded_test = [[x_char_encoder.encode(x_encoder.index_to_token[word]) for word in text] for text in
                           x_padded_test]
    outer_list = []
    for lst in x_char_encoded_test:
        inner_list = []
        for ten in lst:
            res = torch.zeros(MAX_WORD_LENGTH, dtype=torch.long)
            res[:ten.shape[0]] = ten
            inner_list.append(res)
        outer_list.append(inner_list)

    x_char_padded_test = torch.stack([torch.stack(lst) for lst in outer_list])

    return x_char_encoder, x_char_padded_train, x_char_padded_test


def tokenize_pos_tags(X_tags_train, X_tags_test):
    x_postag_encoder=StaticTokenizerEncoder(sample=X_tags_train,
                                     append_eos=False,
                                     tokenize=lambda x: x,
                                     )
    x_postag_encoded_train = [x_postag_encoder.encode(text) for text in X_tags_train]
    x_postag_padded_train = torch.LongTensor(pad_sequence(x_postag_encoded_train, MAX_SENTENCE_LEN+1))
    #x_postag_ohe_train = torch.nn.functional.one_hot(x_postag_padded_train)

    x_postag_encoded_test = [x_postag_encoder.encode(text) for text in X_tags_test]
    x_postag_padded_test = torch.LongTensor(pad_sequence(x_postag_encoded_test, MAX_SENTENCE_LEN+1))

    if x_postag_padded_train.shape[1] > x_postag_padded_test.shape[1]:
        x_postag_padded_test = torch.cat((x_postag_padded_test, torch.zeros(x_postag_padded_test.shape[0], x_postag_padded_train.shape[1]-x_postag_padded_test.shape[1])), dim=1).type(torch.long)

    #x_postag_ohe_test = torch.nn.functional.one_hot(x_postag_padded_test)
    return x_postag_encoder, x_postag_padded_train, x_postag_padded_test

def encode_ner_y(y_ner_list_train, y_ner_list_test, CLASS_COUNT_DICT):
    y_ner_encoder = LabelEncoder(sample=CLASS_COUNT_DICT.keys())
    y_ner_encoded_train = [[y_ner_encoder.encode(label) for label in label_list] for label_list in y_ner_list_train]
    y_ner_encoded_train = [torch.stack(tens) for tens in y_ner_encoded_train]
    y_ner_padded_train = torch.LongTensor(pad_sequence(y_ner_encoded_train, MAX_SENTENCE_LEN + 1))

    y_ner_encoded_test = [[y_ner_encoder.encode(label) for label in label_list] for label_list in y_ner_list_test]
    y_ner_encoded_test = [torch.stack(tens) for tens in y_ner_encoded_test]
    y_ner_padded_test = torch.LongTensor(pad_sequence(y_ner_encoded_test, MAX_SENTENCE_LEN + 1))

    if y_ner_padded_train.shape[1] > y_ner_padded_test.shape[1]:
        y_ner_padded_test = torch.cat((y_ner_padded_test, torch.zeros(y_ner_padded_test.shape[0],
                                                                      y_ner_padded_train.shape[1] -
                                                                      y_ner_padded_test.shape[1])), dim=1).type(
            torch.long)

    return y_ner_padded_train, y_ner_padded_test

#Sample weights
def calculate_sample_weights(y_ner_padded_train):
    ner_class_weights = compute_class_weight('balanced',
                                             classes=np.unique(torch.flatten(y_ner_padded_train).numpy()),
                                             y=torch.flatten(y_ner_padded_train).numpy())

    return ner_class_weights

#Model defintion
#Build Model
class EntityExtraction(nn.Module):

    def __init__(self, num_classes, rnn_hidden_size=512, rnn_stack_size=2, rnn_bidirectional=True, word_embed_dim=124,
                 tag_embed_dim=124, char_embed_dim=124, rnn_embed_dim=512,
                 char_embedding=True, dropout_ratio=0.3):
        super().__init__()
        # self variables
        self.NUM_CLASSES = num_classes
        self.word_embed_dim = word_embed_dim
        self.tag_embed_dim = tag_embed_dim
        self.char_embed_dim = char_embed_dim
        self.rnn_embed_dim = rnn_embed_dim
        self.dropout_ratio = dropout_ratio
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_stack_size = rnn_stack_size
        self.rnn_bidirectional = rnn_bidirectional

        # Embedding Layers
        self.word_embed = nn.Embedding(num_embeddings=x_encoder.vocab_size,
                                       embedding_dim=self.word_embed_dim)
        self.word_embed_drop = nn.Dropout(self.dropout_ratio)

        self.char_embed = nn.Embedding(num_embeddings=x_char_encoder.vocab_size,
                                       embedding_dim=self.char_embed_dim)
        self.char_embed_drop = nn.Dropout(self.dropout_ratio)

        self.postag_embed = nn.Embedding(num_embeddings=x_postag_encoder.vocab_size,
                                         embedding_dim=self.tag_embed_dim)
        self.tag_embed_drop = nn.Dropout(self.dropout_ratio)

        # CNN for character input
        self.conv_char = nn.Conv1d(in_channels=self.char_embed_dim, out_channels=52, kernel_size=3, padding=1)
        # self.maxpool_char = nn.MaxPool1d(kernel_size=3)

        # LSTM for concatenated input
        self.lstm_ner = nn.LSTM(input_size=5760,
                                hidden_size=self.rnn_hidden_size,
                                num_layers=self.rnn_stack_size,
                                batch_first=True,
                                dropout=self.dropout_ratio,
                                bidirectional=self.rnn_bidirectional)
        self.lstm_ner_drop = nn.Dropout(self.dropout_ratio)

        # Linear layers
        self.linear1 = nn.Linear(in_features=1024, out_features=512)
        self.linear_drop = nn.Dropout(self.dropout_ratio)
        self.linear_ner = nn.Linear(in_features=512, out_features=self.NUM_CLASSES + 1)  # +1 for padding 0

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
        char_out = char_out.view(char_out_shape[0], char_out_shape[1] * char_out_shape[2], char_out_shape[3])
        char_out = self.conv_char(char_out.permute(0, 2, 1))
        char_out = char_out.view(char_out_shape[0], char_out_shape[1], -1)

        concat = torch.cat((word_out, char_out, tag_out), dim=2)
        concat = F.relu(concat)

        # NER LSTM
        ner_lstm_out, _ = self.lstm_ner(concat)
        ner_lstm_out = self.lstm_ner_drop(ner_lstm_out)

        # Linear
        ner_out = self.linear1(ner_lstm_out)
        ner_out = self.linear_drop(ner_out)

        # Final Linear
        ner_out = self.linear_ner(ner_out)

        return ner_out



class ClassificationModelUtils:
    def __init__(self, dataloader_train, dataloader_test, ner_class_weights, cuda=True, dropout=0.3, rnn_stack_size=2, learning_rate=0.001):
        if cuda:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            torch.cuda.empty_cache()
        else:
            self.device = torch.device('cpu')

        self.learning_rate = learning_rate

        self.model = EntityExtraction(num_classes=NUM_CLASSES, dropout_ratio=dropout, rnn_stack_size=rnn_stack_size)
        self.model = self.model.to(self.device)

        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test

        self.ner_class_weights = ner_class_weights

        self.criterion_crossentropy = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.ner_class_weights).to(device))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Train metric result holders
        self.epoch_losses = []
        self.epoch_ner_accuracy = []
        self.epoch_ner_recall = []
        self.epoch_ner_precision = []
        self.epoch_ner_f1s = []

        # Test metric result holders
        self.test_epoch_loss = []
        self.test_epoch_ner_accuracy = []
        self.test_epoch_ner_recall = []
        self.test_epoch_ner_precision = []
        self.test_epoch_ner_f1s = []

        # CRF
        self.crf_model = CRF(13).to(device)

    def evaluate_classification_metrics(self, truth, prediction, type='ner'):
        if type == 'ner':
            average = 'macro'
        else:
            average = None
        precision = precision_score(truth, prediction, average=average)
        accuracy = accuracy_score(truth, prediction)
        f1 = f1_score(truth, prediction, average=average)
        recall = recall_score(truth, prediction, average=average)
        return accuracy, precision, recall, f1

    def plot_graphs(self, figsize=(24, 22)):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(3, 2, 2)
        ax.plot(self.epoch_losses, color='b', label="Train")
        ax.plot(self.test_epoch_loss, color='g', label="Test")
        ax.legend()
        ax.set_title("Loss")

        ax = fig.add_subplot(3, 2, 3)
        ax.plot(self.epoch_ner_accuracy, color='b', label="Train")
        ax.plot(self.test_epoch_ner_accuracy, color='g', label="Test")
        ax.legend()
        ax.set_title("Accuracy")

        ax = fig.add_subplot(3, 2, 4)
        ax.plot(self.epoch_ner_precision, color='b', label="Train")
        ax.plot(self.test_epoch_ner_precision, color='g', label="Test")
        ax.legend()
        ax.set_title("Precision")

        ax = fig.add_subplot(3, 2, 5)
        ax.plot(self.epoch_ner_recall, color='b', label="Train")
        ax.plot(self.test_epoch_ner_recall, color='g', label="Test")
        ax.legend()
        ax.set_title("Recall")

        ax = fig.add_subplot(3, 2, 6)
        ax.plot(self.epoch_ner_f1s, color='b', label="Train")
        ax.plot(self.test_epoch_ner_f1s, color='g', label="Test")
        ax.legend()
        ax.set_title("F1")

        plt.show()

    def validate(self):
        test_losses = []
        test_ner_accs = []
        test_ner_precisions = []
        test_ner_recalls = []
        test_ner_f1s = []

        self.test_epoch_prediction_all = []
        self.test_epoch_truth_all = []

        print("************Evaluating validation data now***************")
        for k, data_test in enumerate(self.dataloader_test):
            with torch.no_grad():

                data_test['x_padded'] = data_test['x_padded'].to(self.device)
                data_test['x_char_padded'] = data_test['x_char_padded'].to(self.device)
                data_test['x_postag_padded'] = data_test['x_postag_padded'].to(self.device)
                data_test['y_ner_padded'] = data_test['y_ner_padded'].to(self.device)

                test_ner_out = self.model(data_test['x_padded'], data_test['x_char_padded'],
                                                           data_test['x_postag_padded'])

                # Loss
                #test_loss = self.criterion_crossentropy(test_ner_out.transpose(2, 1), data_test['y_ner_padded'])
                test_loss = -1 * self.crf_model(test_ner_out.permute(1, 0, 2), data_test['y_ner_padded'].permute(1, 0))
                test_losses.append(test_loss.item())

                # Evaluation Metrics
                test_ner_out_result = torch.flatten(torch.argmax(test_ner_out, dim=2)).to('cpu').numpy()
                test_ner_truth_result =torch.flatten(data_test['y_ner_padded']).to('cpu').numpy()

                _ = [self.test_epoch_prediction_all.append(out) for out in test_ner_out_result]
                _ = [self.test_epoch_truth_all.append(out) for out in test_ner_truth_result]


                test_ner_accuracy, test_ner_precision, test_ner_recall, test_ner_f1 = self.evaluate_classification_metrics(
                    self.test_epoch_truth_all, self.test_epoch_prediction_all)

                test_ner_accs.append(test_ner_accuracy)
                test_ner_precisions.append(test_ner_precision)
                test_ner_recalls.append(test_ner_recall)
                test_ner_f1s.append(test_ner_f1)

        self.test_epoch_loss.append(np.array(test_losses).mean())

        self.test_epoch_ner_accuracy.append(test_ner_accuracy)
        self.test_epoch_ner_precision.append(test_ner_precision)
        self.test_epoch_ner_recall.append(test_ner_recall)
        self.test_epoch_ner_f1s.append(test_ner_f1)

        print(f"-->Validation Loss - {self.test_epoch_loss[-1]:.4f}, "
              f"Validation Accuracy - {self.test_epoch_ner_accuracy[-1]} "
              f"Validation Precision - {self.test_epoch_ner_precision[-1]}, "
              f"Validation Recall - {self.test_epoch_ner_recall[-1]} "+
              f"Validation F1 - {self.test_epoch_ner_f1s[-1]}")

    def train(self, num_epochs=10):
        index_metric_append = int(len(dataloader_train) / 4)
        for epoch in range(num_epochs):
            print(f"\n\n------------------------- Epoch - {epoch + 1} -------------------------")
            batch_losses = []
            batch_ner_accuracy = []
            batch_ner_f1s = []
            batch_ner_recalls = []
            batch_ner_precisions = []

            self.epoch_prediction_all = []
            self.epoch_truth_all = []

            for batch_num, data in enumerate(dataloader_train):
                self.optimizer.zero_grad()

                data['x_padded'] = data['x_padded'].to(self.device)
                data['x_char_padded'] = data['x_char_padded'].to(self.device)
                data['x_postag_padded'] = data['x_postag_padded'].to(self.device)
                data['y_ner_padded'] = data['y_ner_padded'].to(self.device)

                ner_out = self.model(data['x_padded'],
                                     data['x_char_padded'],
                                     data['x_postag_padded'])

                # Loss
                #loss = self.criterion_crossentropy(ner_out.transpose(2, 1), data['y_ner_padded'])
                loss = -1*self.crf_model(ner_out.permute(1,0,2), data['y_ner_padded'].permute(1,0))
                batch_losses.append(loss.item())

                # Evaluation Metrics
                test_ner_out_result = torch.flatten(torch.argmax(ner_out, dim=2)).to('cpu').numpy()
                test_ner_truth_result = torch.flatten(data['y_ner_padded']).to('cpu').numpy()

                _ = [self.epoch_prediction_all.append(out) for out in test_ner_out_result]
                _ = [self.epoch_truth_all.append(out) for out in test_ner_truth_result]

                ner_accuracy, ner_precision, ner_recall, ner_f1 = self.evaluate_classification_metrics(self.epoch_truth_all, self.epoch_prediction_all)

                batch_ner_accuracy.append(ner_accuracy)
                batch_ner_precisions.append(ner_precision)
                batch_ner_recalls.append(ner_recall)
                batch_ner_f1s.append(ner_f1)


                if batch_num % index_metric_append == 0 and batch_num != 0:
                    print(f"--> Batch - {batch_num + 1}, " +
                          f"Loss - {np.array(batch_losses).mean():.4f}, " +
                          f"Accuracy - {ner_accuracy:.2f}, " +
                          f"Precision - {ner_precision:.2f}, " +
                          f"Recall - {ner_recall:.2f}, " +
                          f"F1 - {ner_f1:.2f}")

                loss.backward()
                self.optimizer.step()

            self.epoch_losses.append(np.array(batch_losses).mean())

            self.epoch_ner_accuracy.append(ner_accuracy)
            self.epoch_ner_precision.append(ner_precision)
            self.epoch_ner_recall.append(ner_recall)
            self.epoch_ner_f1s.append(ner_f1)

            self.validate()
            print(classification_report(self.test_epoch_truth_all, self.test_epoch_prediction_all))
            #self.plot_graphs()



if __name__ == "__main__":
    EPOCHS = 30
    DROPOUT = 0.5
    RNN_STACK_SIZE = 2
    LEARNING_RATE = 0.0001
    mlflow.set_experiment("PytorchDualLoss")
    with mlflow.start_run() as run:
        mlflow.log_param("Type", "WORD-CHAR-POS-CNN-RNN-CRF-NER-CROSSENTROPY-LOSS")
        mlflow.log_param("EPOCHS", EPOCHS)
        mlflow.log_param("DROPOUT", DROPOUT)
        mlflow.log_param("RNN_STACK_SIZE", RNN_STACK_SIZE)
        mlflow.log_param("LEARNING_RATE", LEARNING_RATE)
        # Load Data
        X_text_list, y_ner_list = load_data('data/dataset_ready.pkl')

        # Get POS tags
        X_tags = get_POS_tags(X_text_list)

        # Split data in test and train plus return segregate as input lists
        X_text_list_train, X_text_list_test, X_tags_train, X_tags_test, \
        y_ner_list_train, y_ner_list_test = split_test_train(X_text_list, X_tags, y_ner_list, split_size=0.3)

        # Set some important parameters values
        MAX_SENTENCE_LEN = max([len(sentence) for sentence in X_text_list_train])
        ALL_LABELS = []
        _ = [[ALL_LABELS.append(label) for label in lst] for lst in y_ner_list_train]
        CLASS_COUNT_OUT = np.unique(ALL_LABELS, return_counts=True)
        CLASS_COUNT_DICT = dict(zip(CLASS_COUNT_OUT[0], CLASS_COUNT_OUT[1]))
        NUM_CLASSES = len([clas for clas in CLASS_COUNT_DICT.keys()])
        print(F"Max sentence length - {MAX_SENTENCE_LEN}, Total Classes = {NUM_CLASSES}")

        mlflow.log_param("MAX_SENTENCE_LEN", MAX_SENTENCE_LEN)
        mlflow.log_param("NUM_CLASSES", NUM_CLASSES)


        # Tokenize Sentences
        x_encoder, x_padded_train, x_padded_test = tokenize_sentence(X_text_list_train, X_text_list_test, MAX_SENTENCE_LEN)

        # Tokenize Characters
        x_char_encoder, x_char_padded_train, x_char_padded_test = tokenize_character(X_text_list_train, x_padded_train, x_padded_test, x_encoder)

        # Tokenize Pos tags
        x_postag_encoder, x_postag_padded_train, x_postag_padded_test = tokenize_pos_tags(X_tags_train, X_tags_test)

        # Encode y NER
        y_ner_padded_train, y_ner_padded_test = encode_ner_y(y_ner_list_train, y_ner_list_test, CLASS_COUNT_DICT)

        #Create train dataloader
        dataset_train = Dataset([{'x_padded': x_padded_train[i],
                                  'x_char_padded': x_char_padded_train[i],
                                  'x_postag_padded': x_postag_padded_train[i],
                                  'y_ner_padded': y_ner_padded_train[i],
                                  } for i in range(x_padded_train.shape[0])])

        dataloader_train = DataLoader(dataset=dataset_train, batch_size=512, shuffle=True)


        # Create test dataloader
        dataset_test = Dataset([{'x_padded': x_padded_test[i],
                                 'x_char_padded': x_char_padded_test[i],
                                 'x_postag_padded': x_postag_padded_test[i],
                                 'y_ner_padded': y_ner_padded_test[i],
                                 } for i in range(x_padded_test.shape[0])])

        dataloader_test = DataLoader(dataset=dataset_test, batch_size=512, shuffle=False)

        #Build model
        GPU = True
        mlflow.log_param("CUDA", GPU)

        ner_class_weights = calculate_sample_weights(y_ner_padded_train)

        model_utils = ClassificationModelUtils(dataloader_train, dataloader_test, ner_class_weights, cuda=GPU, rnn_stack_size=RNN_STACK_SIZE)
        model_utils.train(EPOCHS)
        mlflow.pytorch.log_model(model_utils.model, 'models')

        mlflow.log_metric("Loss-Test", model_utils.test_epoch_loss[-1])
        mlflow.log_metric("Loss-Train", model_utils.epoch_losses[-1])

        mlflow.log_metric("Accuracy-Test", model_utils.test_epoch_ner_accuracy[-1])
        mlflow.log_metric("Accuracy-Train", model_utils.epoch_ner_accuracy[-1])

        mlflow.log_metric("Precision-Test", model_utils.test_epoch_ner_precision[-1])
        mlflow.log_metric("Precision-Train", model_utils.epoch_ner_precision[-1])

        mlflow.log_metric("Recall-Test", model_utils.test_epoch_ner_recall[-1])
        mlflow.log_metric("Recall-Train", model_utils.epoch_ner_recall[-1])

        mlflow.log_metric("F1-Test", model_utils.test_epoch_ner_f1s[-1])
        mlflow.log_metric("F1-Train", model_utils.epoch_ner_f1s[-1])
        model_utils.plot_graphs()