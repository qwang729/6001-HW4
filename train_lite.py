#!/usr/bin/env python3
"""
Text Sentiment Classification using Custom LSTM Model
- Memory efficient implementation
- No external packages beyond torch, numpy, pandas
- Custom vocabulary-based embeddings
"""

import warnings
warnings.filterwarnings('ignore')
path_prefix = './'

import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import re
import os
from collections import Counter
from torch.utils import data

pattern = r'\w+|[^\w\s]'

# ============== Data Loading Functions ==============
def load_training_data(path='train.csv'):
    data = pd.read_csv(path).to_numpy()
    lines = data[:,0]
    x = [re.findall(pattern, line) for line in lines]
    if 'unlabel' not in path:
        y = data[:,1]
        return x, y
    else:
        return x

def load_testing_data(path='test.csv'):
    data = pd.read_csv(path).to_numpy()
    lines = data[:,1]
    x = [re.findall(pattern, line) for line in lines]
    return x

def evaluation(outputs, labels):
    outputs[outputs>=0.5] = 1
    outputs[outputs<0.5] = 0
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

# ============== Simple Vocabulary-Based Embedding ==============
class SimpleVocabEmbedding:
    def __init__(self, sentences, vector_size=250, min_count=2):
        self.vector_size = vector_size
        self.min_count = min_count
        self.wv = {}
        self._build_vocab(sentences)
    
    def _count_words(self, sentences):
        word_counts = Counter()
        for sentence in sentences:
            word_counts.update(sentence)
        return word_counts
    
    def _build_vocab(self, sentences):
        print("Building vocabulary...")
        word_counts = self._count_words(sentences)
        vocab = [word for word, count in word_counts.items() if count >= self.min_count]
        print(f"Vocabulary size: {len(vocab)} (min_count={self.min_count})")
        for word in vocab:
            self.wv[word] = np.random.randn(self.vector_size).astype(np.float32) * 0.1
        print("Embeddings initialized.")
    
    def save(self, path):
        torch.save(self.wv, path)
    
    @classmethod
    def load(cls, path):
        model = cls.__new__(cls)
        model.wv = torch.load(path, weights_only=False)
        return model


# ============== Preprocessing ==============
class Preprocess():
    def __init__(self, sentences, sen_len, w2v_path="./w2v.model"):
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []
    
    def get_w2v_model(self):
        self.wv_dict = torch.load(self.w2v_path, weights_only=False)
        self.embedding_dim = len(list(self.wv_dict.values())[0])
    
    def add_embedding(self, word):
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)
    
    def make_embedding(self, load=True):
        print("Get embedding ...")
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        for i, word in enumerate(self.wv_dict):
            print('get words #{}'.format(i+1), end='\r')
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(torch.tensor(self.wv_dict[word], dtype=torch.float32))
        print('')
        self.embedding_matrix = torch.stack(self.embedding_matrix)
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix
    
    def pad_sequence(self, sentence):
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence
    
    def sentence_word2idx(self):
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                if word in self.word2idx:
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)
    
    def labels_to_tensor(self, y):
        y = [int(label) for label in y]
        return torch.LongTensor(y)


# ============== Dataset ==============
class SenDataset(data.Dataset):
    def __init__(self, X, y):
        self.data = X
        self.label = y
    def __getitem__(self, idx):
        if self.label is None: 
            return self.data[idx]
        return self.data[idx], self.label[idx]
    def __len__(self):
        return len(self.data)


# ============== LSTM Model ==============
class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True, bidirectional=False):
        super(LSTM_Net, self).__init__()
        self.embedding = nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, inputs):
        embedded = self.embedding(inputs)
        lstm_out, (hidden, cell) = self.lstm(embedded, None)
        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        output = self.classifier(hidden)
        return output


# ============== Training Function ==============
def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    
    model.train()
    criterion = nn.BCELoss()
    t_batch = len(train)
    v_batch = len(valid)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=False)
    
    total_loss, total_acc, best_acc = 0, 0, 0
    
    for epoch in range(n_epoch):
        model.train()
        total_loss, total_acc = 0, 0
        
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze()
            
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            correct = evaluation(outputs.detach(), labels)
            total_acc += correct
            total_loss += loss.item()
            
            if (i + 1) % 50 == 0:
                print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
                    epoch+1, i+1, t_batch, loss.item(), correct*100/batch_size), end='\r')
        
        train_acc = total_acc / (t_batch * batch_size)
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/t_batch, train_acc*100))
        
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.float)
                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                correct = evaluation(outputs, labels)
                val_acc += correct
                val_loss += loss.item()
            
            val_acc = val_acc / (v_batch * batch_size)
            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(val_loss/v_batch, val_acc*100))
            
            scheduler.step(val_acc)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model, "{}/ckpt.model".format(model_dir))
                print('saving model with acc {:.3f}'.format(val_acc*100))
        
        print('-----------------------------------------------')
    
    return best_acc


# ============== Testing Function ==============
def testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs>=0.5] = 1
            outputs[outputs<0.5] = 0
            ret_output += outputs.int().tolist()
    return ret_output


# ============== Main ==============
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_with_label = os.path.join(path_prefix, 'train.csv')
    train_no_label = os.path.join(path_prefix, 'train_unlabel.csv')
    testing_data = os.path.join(path_prefix, 'test.csv')
    w2v_path = os.path.join(path_prefix, 'w2v_all.model')
    
    # Memory-efficient hyperparameters
    sen_len = 30
    fix_embedding = False
    batch_size = 256
    epoch = 8
    lr = 0.003
    hidden_dim = 128
    num_layers = 1
    dropout = 0.3
    bidirectional = True
    model_dir = path_prefix
    
    print("loading training data ...")
    train_x, y = load_training_data(train_with_label)
    train_x_no_label = load_training_data(train_no_label)
    
    print("loading testing data ...")
    test_x = load_testing_data(testing_data)
    
    print("\n=== Building Vocabulary Embeddings ===")
    all_sentences = train_x + train_x_no_label + test_x
    w2v_model = SimpleVocabEmbedding(all_sentences, vector_size=250, min_count=2)
    w2v_model.save(w2v_path)
    print(f"Embeddings saved to {w2v_path}\n")
    
    print("\n=== Preprocessing Labeled Data ===")
    preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    train_x_tensor = preprocess.sentence_word2idx()
    y_tensor = preprocess.labels_to_tensor(y)
    
    split_idx = int(len(train_x_tensor) * 0.9)
    X_train, X_val = train_x_tensor[:split_idx], train_x_tensor[split_idx:]
    y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    train_dataset = SenDataset(X=X_train, y=y_train)
    val_dataset = SenDataset(X=X_val, y=y_val)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=hidden_dim, num_layers=num_layers, 
                     dropout=dropout, fix_embedding=fix_embedding, bidirectional=bidirectional)
    model = model.to(device)
    
    print("\n=== Initial Training ===")
    best_acc = training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)
    
    print("\n=== Final Prediction ===")
    preprocess_test = Preprocess(test_x, sen_len, w2v_path=w2v_path)
    _ = preprocess_test.make_embedding(load=True)
    test_x_tensor = preprocess_test.sentence_word2idx()
    
    test_dataset = SenDataset(X=test_x_tensor, y=None)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print('load model ...')
    model = torch.load(os.path.join(model_dir, 'ckpt.model'), weights_only=False)
    outputs = testing(batch_size, test_loader, model, device)
    
    tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))], "labels":outputs})
    print("save csv ...")
    tmp.to_csv(os.path.join(path_prefix, 'predict.csv'), index=False)
    print("Finish Predicting")

if __name__ == "__main__":
    main()
