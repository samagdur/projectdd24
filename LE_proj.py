import pandas as pd
import csv
import re
import os
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
import pickle
import gdown
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gdown # too large, calab can not do virus controll, must use this
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_size = 32):
        super(LogisticRegressionModel, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x

def train_logistic_regression(model, criterion, optimizer, X_train, y_train, num_epochs=350, batch_size=128, patience=5):
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
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {running_loss}')

def evaluate_model(model, X_test, y_test):
    model.eval()
    y_test = y_test.cpu().numpy()
    with torch.no_grad():
        outputs = model(torch.Tensor(X_test)).cpu().numpy()
        predicted_labels = (outputs >= 0.5).astype(int).squeeze()

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

def evaluate_transformer_model(model, val_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
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
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

def get_data(old):
  if old:
    import pickle
    with open('glove_embedding.pkl', 'rb') as pickle_file:
        glove_embeddings = pickle.load(pickle_file)
    #with open('data.pkl', 'rb') as pickle_file:
    #    data = pickle.load(pickle_file)
    with open('labels.pkl', 'rb') as pickle_file:
        labels = pickle.load(pickle_file)
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

    labels = data[1:,5].astype(int)
    print(f'We have {np.sum(labels)} positive labels out of {labels.shape[0]} that is good!\n')

    question1 = data[1:,3]
    question2 = data[1:,4]

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
    #with open('data.pkl', 'wb') as f:
    #    pickle.dump(data, f)
    import pickle
    with open('labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    with open('question1.pkl', 'wb') as f:
        pickle.dump(question1, f)
    with open('question2.pkl', 'wb') as f:
        pickle.dump(question2, f)
    print("Size of the array:", data.nbytes / (10**9), "GB")
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
    #print(glove_embeddings['the'])
  return glove_embeddings, labels, question1, question2

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
                pass
                #print(word)
        sequences.append(embedding_sequence)
    return sequences

def pad_sequences_with_length(sequences, max_length):
    return pad_sequences(sequences, maxlen=max_length, padding='post',
                        truncating='post', dtype='float32')

def reduce_data(size, old=False, max_length = 22):
  glove_embeddings, labels, question1, question2 = get_data(old=old)

  question1_sequences = sentences_to_sequences(question1, glove_embeddings)
  question2_sequences = sentences_to_sequences(question2, glove_embeddings)

  #max_length = max(len(x) for x in question1_sequences)
  #print('The max length: ',max_length)

  question1_padded = pad_sequences_with_length(question1_sequences, max_length)
  question2_padded = pad_sequences_with_length(question2_sequences, max_length)

  lengths = []
  for q in [question1, question2]:
    for line in q:
      lengths.append(len(line.split()))
  plt.figure(figsize=(10, 6))
  plt.hist(lengths, bins=range(51), color='skyblue', edgecolor='black')
  plt.title('Histogram of Sentence Lengths (Capped at 50)')
  plt.xlabel('Sentence Length')
  plt.ylabel('Frequency')
  plt.xticks(range(0, 51, 5))
  plt.grid(axis='y', linestyle='--', alpha=0.7)
  plt.show()

  del question1_sequences
  del question2_sequences
  del question1
  del question2
  del glove_embeddings

  # select even dataset
  n_pos = size#140000# 149 000 is max
  count_pos, count_neg = 0, 0
  ind = []
  for i, el in enumerate(labels):
    if el ==1 and count_pos < n_pos:
        ind.append(i)
        count_pos+=1
    if el ==0 and count_neg < n_pos:
      ind.append(i)
      count_neg+=1
  labels_selected = labels[ind]

  question1_padded = question1_padded[ind, :, :]
  question2_padded = question2_padded[ind, :, :]
  y = labels[ind]
  X = np.concatenate((question1_padded, question2_padded), axis=1)

  first_dim = X.shape[1]
  second_dim = X.shape[2]

  X = X.reshape((X.shape[0], first_dim * second_dim))
  del question1_padded
  del question2_padded
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, random_state=42) # test split
  del X
  import torch
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.12, random_state=42) # validation split

  X_train = torch.tensor(X_train, dtype=torch.float32)
  X_test = torch.tensor(X_test, dtype=torch.float32)
  X_val = torch.tensor(X_val, dtype=torch.float32)

  y_train = torch.tensor(y_train, dtype=torch.float32)
  y_test = torch.tensor(y_test, dtype=torch.float32)
  y_val = torch.tensor(y_val, dtype=torch.float32)


  return X_train, X_val, X_test, y_train, y_val, y_test, (first_dim, second_dim)

def eval(y_true, y_pred):
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


def get_data_easy():
    import os
    if not os.path.isfile('data_new.pkl'):
      data_new = reduce_data(150000, old=False, max_length = 22)
      with open('data_new.pkl', 'wb') as f:
        pickle.dump(data_new, f)
      with open('data_new.pkl', 'rb') as pickle_file:
        data_new = pickle.load(pickle_file)
    return data_new



# =====================RUN HERE=======================================
if torch.cuda.is_available():
  device = torch.device("cuda")
  print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
  device = torch.device("cpu")
  print("GPU not available, using CPU")

def logistic():
  X_train, X_val, X_test, y_train, y_val, y_test, (first_dim, second_dim)= get_data_easy()
  # logistic
  input_dim = X_train.shape[1]
  logistic_regression_model = LogisticRegressionModel(input_dim=input_dim,
                                                      hidden_size=32).to(device)
  criterion = nn.BCELoss()
  lr = 1e-3
  optimizer_lr = torch.optim.Adam(logistic_regression_model.parameters(), lr=lr)
  train_logistic_regression(logistic_regression_model, criterion, optimizer_lr,
                            X_train.to(device), y_train.to(device))
  evaluate_model(logistic_regression_model, X_test.to(device), y_test.to(device))

  return logistic_regression_model

def transformer():
  X_train, X_val, X_test, y_train, y_val, y_test, (first_dim, second_dim)= get_data_easy()
  X_train = X_train.reshape(X_train.shape[0], first_dim, second_dim)
  X_test = X_test.reshape(X_test.shape[0], first_dim, second_dim)


  train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
  test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

  input_dim = X_train.shape[2]
  hidden_dim = 128
  n_layers = 2
  n_heads = 2
  dropout_prob = 0.05
  lr = 0.085
  batch_size = 64
  num_epochs = 20

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size)

  transformer_model = TransformerModel(input_dim, hidden_dim, n_layers, n_heads, dropout_prob).to(device)

  criterion = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(transformer_model.parameters(), lr=lr)

  train_transformer(transformer_model, criterion, optimizer, train_loader,
                    test_loader, num_epochs)


  evaluate_transformer_model(transformer_model, test_loader)
#transformer()

def cosine():
  import torch.nn.functional as F
  X_train, X_val, X_test, y_train, y_val, y_test, (first_dim, second_dim)= get_data_easy()
  X_train = X_train.reshape(X_train.shape[0], 2, int(first_dim/2), second_dim)# add to one vector

  summed_represent = torch.sum(X_train, dim=2)
  cosine_sim = F.cosine_similarity(summed_represent[:, 0, :], summed_represent[:, 1, :], dim=1)
  cos_val = cosine_sim.numpy()
  plt.figure(figsize=(10, 6))
  plt.hist(cos_val, bins=30, range=(0.4, 1.0), color='blue', edgecolor='black')
  plt.title('Histogram of Cosine Similarity (Range: 0.4 to 1.0)')
  plt.xlabel('Cosine Similarity')
  plt.ylabel('Frequency')
  plt.show()

  print(cosine_sim[:10])
  best_acc = 0
  best_par = None
  acc_list = []
  x = np.linspace(0.001,0.999, 300)
  y_train = y_train.cpu().numpy()
  for lim in x:
    pred_cos = np.zeros_like(cos_val)
    pred_cos[cos_val >= lim] =1
    pred_cos[cos_val < lim] =0
    acc = np.sum(1-np.abs(pred_cos-y_train))/pred_cos.shape[0]
    acc_list.append(acc)
    if acc > best_acc:
      best_acc = acc
      best_par = lim
  print('Best accuracy and cutoff-value',best_acc, best_par)
  plt.figure(figsize=(10, 5))
  plt.plot(x, acc_list, marker='.', linestyle='-', color='b')
  plt.title('Hyperparameter Search: Accuracy vs. Cutoff Value')
  plt.xlabel('Cutoff Value')
  plt.ylabel('Accuracy')
  plt.show()

  lim = best_par
  pred_cos = np.zeros_like(cos_val)
  pred_cos[cos_val >= lim] =1
  pred_cos[cos_val < lim] =0
  eval(y_train, pred_cos)
  #test
  X_test= X_test.reshape(X_test.shape[0], 2, int(first_dim/2), second_dim)# add to one vector
  summed_represent = torch.sum(X_test, dim=2)
  cosine_sim = F.cosine_similarity(summed_represent[:, 0, :], summed_represent[:, 1, :], dim=1)
  cos_val = cosine_sim.numpy()
  lim = best_par
  pred_cos = np.zeros_like(cos_val)
  pred_cos[cos_val >= lim] =1
  pred_cos[cos_val < lim] =0
  eval(y_test, pred_cos)
#cosine()
def get_raw():
    if not os.path.isfile('labels_raw.pkl'):
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
            pass
            print(proc)
          if len(proc) ==6: #257 / 404303 are broken, just do not add those.
            out.append(proc)
      data = np.array(out)
      labels = data[1:,5].astype(int)
      question1 = data[1:,3]
      question2 = data[1:,4]

      n_pos = 140000# 149 000 is max
      count_pos, count_neg = 0, 0
      ind = []
      for i, el in enumerate(labels):
        if el ==1 and count_pos < n_pos:
            ind.append(i)
            count_pos+=1
        if el ==0 and count_neg < n_pos:
          ind.append(i)
          count_neg+=1
      print('len',len(ind))

      labels = labels[ind]
      question1 = question1[ind]
      question2 = question2[ind]
      import pickle
      with open('labels_raw.pkl', 'wb') as f:
          pickle.dump(labels, f)
      with open('question1_raw.pkl', 'wb') as f:
          pickle.dump(question1, f)
      with open('question2_raw.pkl', 'wb') as f:
          pickle.dump(question2, f)
    import pickle
    with open('labels_raw.pkl', 'rb') as pickle_file:
        labels = pickle.load(pickle_file)
    with open('question1_raw.pkl', 'rb') as pickle_file:
        question1 = pickle.load(pickle_file)
    with open('question2_raw.pkl', 'rb') as pickle_file:
        question2 = pickle.load(pickle_file)
    return labels, question1, question2


def transformer2():
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
  if os.path.isfile('model.pth'):
    model.load_state_dict(torch.load('model.pth'))

  labels, question1, question2 = get_raw()
  labels, question1, question2 = labels[10000:], question1[10000:], question2[10000:]
  labels_test, question1_test, question2_test = labels[:10000], question1[:10000], question2[:10000]

  print('Fraction of positive in train',np.sum(labels)/labels.shape[0])
  print('Fraction of positive in test', np.sum(labels_test)/labels_test.shape[0])
  num_epochs = 5
  lr = 0.01
  batch_size = 100
  criterion = nn.CrossEntropyLoss()

  optimizer = optim.Adam(model.parameters(), lr=lr)
  encoding = tokenizer(question1.tolist(), question2.tolist(),
                     padding=True, truncation=True,
                     return_tensors="pt", max_length=100)
  train_dataset = TensorDataset(encoding['input_ids'], encoding['attention_mask'], torch.Tensor(labels).long())
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  for epoch in range(num_epochs):
    model = model.to(device)
    model.train()
    train_loss = 0.0
    for input_ids, attention_mask, labels in train_loader:
        input_ids= input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
    torch.save(model.state_dict(), 'model.pth')
    del attention_mask
    del input_ids
    del labels
  #test
  torch.cuda.empty_cache()
  encoding = tokenizer(question1_test.tolist(), question2_test.tolist(),
                     padding=True, truncation=True,
                     return_tensors="pt", max_length=100)
  test_dataset = TensorDataset(encoding['input_ids'], encoding['attention_mask'], torch.Tensor(labels_test).long())
  test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
  preds = []
  with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
      input_ids= input_ids.to(device)
      attention_mask = attention_mask.to(device)
      labels = labels.to(device)

      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      logits = outputs.logits
      probs = torch.nn.functional.softmax(logits, dim=1)
      eq = probs[:, 1] > probs[:, 0]
      #print(1-(np.sum(np.abs(eq.cpu().numpy().astype(int)-labels_test)))/labels_test.shape[0])
      preds.append(eq.cpu().numpy())

  eval(np.concatenate(preds), labels_test)

def transformer3():
  from transformers import BertModel, BertTokenizer
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertModel.from_pretrained('bert-base-uncased')

  if not os.path.isfile('labels.pkl'):
    labels, question1, question2 = get_raw()
    encoding1 = tokenizer(question1.tolist(), padding=True, truncation=True, return_tensors="pt", max_length=100)
    encoding2 = tokenizer(question2.tolist(), padding=True, truncation=True,
                        return_tensors="pt", max_length=100)
    with open('encoding1.pkl', 'wb') as f:
        pickle.dump(encoding1, f)
    with open('encoding2.pkl', 'wb') as f:
      pickle.dump(encoding2, f)
    with open('labels.pkl', 'wb') as f:
      pickle.dump(labels, f)
  else:
    with open('encoding1.pkl', 'rb') as pickle_file:
        encoding1 = pickle.load(pickle_file)
    with open('encoding2.pkl', 'rb') as pickle_file:
        encoding2 = pickle.load(pickle_file)
    with open('labels.pkl', 'rb') as pickle_file:
        labels = pickle.load(pickle_file)
  ids1 = encoding1['input_ids']
  ids2 = encoding2['input_ids']
  atm1 = encoding1['attention_mask']
  atm2 = encoding2['attention_mask']
  print('ids',ids1.shape)
  train_dataset = TensorDataset(ids1, ids2, atm1, atm2)
  train_loader = DataLoader(train_dataset, batch_size=101, shuffle=False)
  preds = []
  lenght = len(train_loader)
  count = 0
  with torch.no_grad():
    for lim in [0.9]:
      for ids1, ids2, atm1, atm2 in train_loader:
        if count ==300:
          break
        count +=1
        print(f'\r{count}/{lenght}')
        outputs1 = model(input_ids=ids1, attention_mask=atm1)
        outputs2 = model(input_ids=ids2, attention_mask=atm2)


        hidden_states1 = outputs1.last_hidden_state
        hidden_states2 = outputs2.last_hidden_state
        print('hidden',hidden_states1.shape)

        sentence_emb1 = torch.sum(hidden_states1, dim=1)
        sentence_emb2 = torch.sum(hidden_states2, dim=1)

        cos_sim = F.cosine_similarity(sentence_emb1, sentence_emb2, dim=1)
        #lim = 0.93
        cos_val = cos_sim.numpy()
        pred_cos = np.zeros_like(cos_val)
        pred_cos[cos_val >= lim] =1
        pred_cos[cos_val < lim] =0
        preds.append(pred_cos)
      print(lim)
      eval(np.concatenate(preds), labels[:100*(count)])
      count =0
      preds =[]

def get_x_input_from_command_line(model, lemmatizer, max_length=22):
    glove_embeddings, _, _, _ = get_data(False)
    stop_words = set(stopwords.words('english'))
    stop_words = set(word for word in stop_words if word not in ['should','what', 'which', 'who', 'whom', 'where', 'when', 'why', 'how'])

    number_of_questions = int(input("Insert how many questions you want to know are semantically equivalent"))

    for i in range(number_of_questions):
        question1 = input("Insert first question: ")
        question2 = input("Insert second question: ")

        # Define a function to clean and process the questions
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


        # max_length = max(len(seq) for seq in question1_sequences + question2_sequences)


        question1_padded = pad_sequences_with_length(question1_sequences, max_length)
        question2_padded = pad_sequences_with_length(question2_sequences, max_length)


        #question1_padded = question1_padded[:round(len(question1_padded) / 10), :, :]
        #question2_padded = question2_padded[:round(len(question2_padded) / 10), :, :]


        X = np.concatenate((question1_padded, question2_padded), axis=1)


        first_dim = X.shape[1]
        second_dim = X.shape[2]
        X = X.reshape((X.shape[0], first_dim * second_dim))


        model.eval()
        with torch.no_grad():
            outputs = model(torch.Tensor(X))
            predicted_labels = (outputs >= 0.5).int().squeeze().numpy().max()

            if predicted_labels == 1:
                print("Yes, the questions are asking the same thing semantically")
            else:
                print("No, the questions are not asking the same thing semantically")

def main():
    cosine()
    model = logistic()
    get_x_input_from_command_line(model=model, lemmatizer=WordNetLemmatizer())
    # transformer() #scratch
    transformer2() #finetune
    transformer3() # hidden



if __name__ == "__main__":
    main()