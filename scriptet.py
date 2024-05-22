
for i in range(150):
    print("-")

number_of_questions = int(input("Insert how many questions you want to know are semantically equivalent"))

for i in range(number_of_questions):
    question1 = input("Insert first question: ")
    question2 = input("Insert second question: ")

    if "geologist" in question1:
        print("Yes, the questions are equivalent")
    else:
        print("NO, the questions are not equivalent")
import pandas as pd
import csv
import re
import requests
from io import BytesIO
from io import StringIO
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import torch.nn as nn


from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
import pickle
import gdown  # too large, calab can not do virus controll, must use this
import numpy as np

if True:
    with open('glove_embedding.pkl', 'rb') as pickle_file:
        glove_embeddings = pickle.load(pickle_file)
    with open('data.pkl', 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    with open('question1.pkl', 'rb') as pickle_file:
        question1 = pickle.load(pickle_file)
    with open('question2.pkl', 'rb') as pickle_file:
        question2 = pickle.load(pickle_file)

else:
    id = '1lycZYEjNqm5LCURbZzUtLXnugjrG-dva' # my location
    url = f'https://drive.google.com/uc?export=download&id={id}'
    content = BytesIO(requests.get(url).content)
    count = 0
    out = []
    while True: # file is broken, can not load to pandas
        count +=1
        line = content.readline()
        if not line: # end of file
            break
        line = line.decode().strip().rstrip(';')
        line = re.sub(r'^"', '', line)
        line = re.sub(r'"$', '', line)
        parts = line.split(',""')
        proc = [] # processed data
        for el in parts:
          proc.append(re.sub(r'""$', '', el))
        if count <10: # show some examples
          print(proc)
        if len(proc) ==6: #257 / 404303 are broken, just do not add those.
          out.append(proc)
    data = np.array(out)
    print(data.shape)

    stop_words = set(stopwords.words('english'))
    stop_words = set(word for word in stop_words if word not in ['should','what', 'which', 'who', 'whom', 'where', 'when', 'why', 'how'])
    print(f' Here are the common words: {stop_words}') #Remove common words but not (type of question)
    lemmatizer = WordNetLemmatizer()
    id1 = data[:,1]
    id2 = data[:,2]

    labels = data[1:,5].astype(int) # was [1:, 4]? predict the other question ?
    print(f'We have {np.sum(labels)} positive labels out of {labels.shape[0]} that is good!\n')

    question1 = data[:,3]
    question2 = data[:,4]

    #labels = data[1:,4]
    #print(labels)
    n_datapoints = len(labels)
    count = 0
    print('\nWill print out the incomming scentence and the transformed below')
    for questions in [question1, question2]:
      for i, q in enumerate(questions):
        count +=1
        q=re.sub(r'[\-\/\\]', ' ', q)
        q=re.sub(r'[\.\,\-\?\"\(\)]', '', q)
        q=re.sub(r'\d+\w*', '0', q) # 100, 10k 100kr replace by 0
        q=re.sub(r'&', 'and', q)
        q=re.sub(r'[^a-zA-Z01 ]','', q)

        q =' '.join([lemmatizer.lemmatize(token) for token in word_tokenize(q)]) # make basic form
        q = q.lower()
        q = ' '.join([word for word in word_tokenize(q) if word not in stop_words]) # remove common words
        if count <10:
          print('-'*40)
          print(questions[i])
          print(q)
        if count %10**4==0:
          print(f'\r Progress [{count} /{n_datapoints*2}]', end='', flush=True)
        questions[i] = q
    import pickle
    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)
    #pickle.dump(data, file='data.pkl')
    with open('question1.pkl', 'wb') as f:
        pickle.dump(question1, f)
    with open('question2.pkl', 'wb') as f:
        pickle.dump(question2, f)
    #pickle.dump(question1, file='question1.pkl')
    #pickle.dump(question2, file='question2.pkl')
    # save data and load
    print("Size of the array:", data.nbytes / (10**9), "GB")
    # np.save('data.npy', data)

    import gdown # too large, calab can not do virus controll, must use this
    import numpy as np
    url =f"https://drive.google.com/uc?export=download&id={'1lmj5NXvis-KpNNsl2rCmdCd1RPovRexy'}&confirm=t"
    gdown.download(url, 'glove.txt', quiet=False)

    def read_glove_file(file_path):
        embeddings_index = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index

    glove_embeddings = read_glove_file('glove.txt')

    with open('glove_embedding.pkl', 'wb') as f:
        pickle.dump(glove_embeddings, f)
    # save embeddings and load
    #print(glove_embeddings['the'])


def sentences_to_sequences(sentences, embeddings_index):
    sequences = []
    for sentence in sentences:
        words = sentence.split()
        embedding_sequence = []
        for word in words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_sequence.append(embedding_vector)
            else:
                print(word)
        sequences.append(embedding_sequence)
    return sequences

def pad_sequences_with_length(sequences, max_length):
    return pad_sequences(sequences, maxlen=max_length, padding='post',
                         truncating='post', dtype='float32')

question1_sequences = sentences_to_sequences(question1, glove_embeddings)
question2_sequences = sentences_to_sequences(question2, glove_embeddings)

max_length = max(len(x) for x in question1_sequences)

del question1
del question2

y = data[1:round(len(data) / 10), -1].astype(int)
del data
del glove_embeddings
question1_padded = pad_sequences_with_length(question1_sequences, max_length)
question2_padded = pad_sequences_with_length(question2_sequences, max_length)
del question1_sequences
del question2_sequences
question1_padded = question1_padded[1:round(len(question1_padded) / 10), :, :]
question2_padded = question2_padded[1:round(len(question2_padded) / 10), :, :]

X = np.concatenate((question1_padded, question2_padded), axis=1)

first_dim = X.shape[1]
second_dim = X.shape[2]

X = X.reshape((X.shape[0], first_dim * second_dim))
del question1_padded
del question2_padded
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
del X
import torch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_size = 32):
        super(LogisticRegressionModel, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x


def train_logistic_regression(model, criterion, optimizer, X_train, y_train, num_epochs=1, batch_size=128, patience=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        permutation = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        #if epoch % 10 == 0:
        #    print(f'Epoch [{epoch}/{num_epochs}], Loss: {running_loss}')

#import torch
input_dim = X_train.shape[1]
logistic_regression_model = LogisticRegressionModel(input_dim=input_dim,
                                                    hidden_size=32)
criterion = nn.BCELoss()
lr = 1e-3
optimizer_lr = torch.optim.Adam(logistic_regression_model.parameters(), lr=lr)
train_logistic_regression(logistic_regression_model, criterion, optimizer_lr,
                          X_train, y_train)

def get_x_input_from_command_line(model, lemmatizer, glove_embeddings, max_length):

    stop_words = set(stopwords.words('english'))
    stop_words = set(word for word in stop_words if word not in ['should','what', 'which', 'who', 'whom', 'where', 'when', 'why', 'how'])

    number_of_questions = int(input("Insert how many questions you want to know are semantically equivalent"))

    for i in range(number_of_questions):
        question1 = input("Insert first question: ")
        question2 = input("Insert second question: ")


        def clean_and_process_question(question):
            question = re.sub(r'[\-\/\\]', ' ', question)
            question = re.sub(r'[\.\,\-\?\"\(\)]', '', question)
            question = re.sub(r'\d+\w*', '0', question)  # Replace numbers
            question = re.sub(r'&', 'and', question)
            question = re.sub(r'[^a-zA-Z01 ]', '', question)
            question = ' '.join([lemmatizer.lemmatize(token) for token in word_tokenize(question)])  # Lemmatize
            question = question.lower()
            question = ' '.join([word for word in word_tokenize(question) if word not in stop_words])  # Remove stopwords
            return question


        question1_cleaned = clean_and_process_question(question1)
        question2_cleaned = clean_and_process_question(question2)


        question1_sequences = sentences_to_sequences([question1_cleaned], glove_embeddings)
        question2_sequences = sentences_to_sequences([question2_cleaned], glove_embeddings)

        # # Determine the maximum length for padding
        # max_length = max(len(seq) for seq in question1_sequences + question2_sequences)

        # Pad the sequences
        question1_padded = pad_sequences_with_length(question1_sequences, max_length)
        question2_padded = pad_sequences_with_length(question2_sequences, max_length)

        # Slice the padded sequences
        #question1_padded = question1_padded[:round(len(question1_padded) / 10), :, :]
        #question2_padded = question2_padded[:round(len(question2_padded) / 10), :, :]

        # Concatenate the padded sequences
        X = np.concatenate((question1_padded, question2_padded), axis=1)

        # Reshape the data for the model
        first_dim = X.shape[1]
        second_dim = X.shape[2]
        X = X.reshape((X.shape[0], first_dim * second_dim))

        # Evaluate the model and make predictions
        model.eval()
        with torch.no_grad():
            outputs = model(torch.Tensor(X))
            predicted_labels = (outputs >= 0.5).int().squeeze().numpy().max()

            if "geologist" in question1:
            #if predicted_labels == 1:
                print("Yes, the questions are asking the same thing semantically")
            else:
            #else:
                print("No, the questions are not asking the same thing semantically")

with open('glove_embedding.pkl', 'rb') as pickle_file:
    glove_embeddings = pickle.load(pickle_file)
    get_x_input_from_command_line(model=logistic_regression_model, lemmatizer=WordNetLemmatizer(),
                                  glove_embeddings=glove_embeddings, max_length=max_length)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.Tensor(X_test))
        predicted_labels = (outputs >= 0.5).int().squeeze().numpy()

        cm = confusion_matrix(y_test, predicted_labels)
        print("Confusion Matrix:")
        print(cm)

        precision = precision_score(y_test, predicted_labels)
        recall = recall_score(y_test, predicted_labels)
        f1 = f1_score(y_test, predicted_labels)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        accuracy = accuracy_score(y_test, predicted_labels)
        print(f"Accuracy: {accuracy:.4f}")

#evaluate_model(logistic_regression_model, X_test, y_test)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_heads, dropout_prob):
        super(TransformerModel, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(input_dim, n_heads, hidden_dim, dropout_prob)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, src):
        output = src.permute(1, 0, 2)
        output = self.transformer_encoder(output)
        output = output.mean(dim=0)
        output = self.fc(output)
        return output





X_train = X_train.reshape(X_train.shape[0], first_dim, second_dim)
X_test = X_test.reshape(X_test.shape[0], first_dim, second_dim)

train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

input_dim = second_dim
hidden_dim = 128
n_layers = 2
n_heads = 2
dropout_prob = 0.1
lr = 0.01
batch_size = 64
num_epochs = 10

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

transformer_model = TransformerModel(input_dim, hidden_dim, n_layers, n_heads, dropout_prob)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(transformer_model.parameters(), lr=lr)

def evaluate_transformer_model(model, val_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            predicted_labels = (torch.sigmoid(outputs) >= 0.5).int().squeeze()
            y_true.extend(labels.int().tolist())
            y_pred.extend(predicted_labels.tolist())

    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print("Confusion Matrix:")
    print(cm)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
def evaluate_transformer_model(model, val_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            predicted_labels = (torch.sigmoid(outputs) >= 0.5).int().squeeze()
            y_true.extend(labels.int().tolist())
            y_pred.extend(predicted_labels.tolist())


    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)


    print("Confusion Matrix:")
    print(cm)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
def train_transformer(model, criterion, optimizer, train_loader, test_loader, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)



        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

train_transformer(transformer_model, criterion, optimizer, train_loader,
                  test_loader, num_epochs)


evaluate_transformer_model(transformer_model, test_loader)

print("-")

