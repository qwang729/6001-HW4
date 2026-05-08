import warnings
warnings.filterwarnings('ignore')
path_prefix = './'

import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import re
import os
from collections import Counter
import random
from torch.utils import data

pattern = r'\w+|[^\w\s]'

def load_training_data(path='train.csv'):
    # Read training data
    data = pd.read_csv(path).to_numpy()
    lines = data[:,0]
    x = [re.findall(pattern,line) for line in lines]
    if 'unlabel' not in path:
        y = data[:,1]
        return x, y
    else:
        return x

def load_testing_data(path='test.csv'):
    # Read testing data
    data = pd.read_csv(path).to_numpy()
    lines = data[:,1]
    x = [re.findall(pattern,line) for line in lines]
    return x

def evaluation(outputs, labels):
    #outputs => probability (float)
    #labels => labels
    outputs[outputs>=0.5] = 1 # Negtive Sentiment
    outputs[outputs<0.5] = 0 # Positive Sentiment
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct


# Custom Word2Vec implementation
class SimpleWord2Vec:
    def __init__(self, sentences, vector_size=250, window=5, min_count=5, epochs=10, lr=0.025):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.lr = lr
        self.wv = {}
        self._train(sentences)
    
    def _count_words(self, sentences):
        word_counts = Counter()
        for sentence in sentences:
            word_counts.update(sentence)
        return word_counts
    
    def _build_vocab(self, word_counts):
        vocab = [word for word, count in word_counts.items() if count >= self.min_count]
        return vocab
    
    def _initialize_weights(self, vocab):
        for word in vocab:
            self.wv[word] = np.random.uniform(-0.5/self.vector_size, 0.5/self.vector_size, self.vector_size).astype(np.float32)
    
    def _train(self, sentences):
        print("Training Word2Vec...")
        word_counts = self._count_words(sentences)
        vocab = self._build_vocab(word_counts)
        print(f"Vocabulary size: {len(vocab)}")
        self._initialize_weights(vocab)
        
        # Create negative sampling table
        neg_table_size = 10000000
        self.neg_table = np.zeros(neg_table_size, dtype=np.int32)
        
        # Build frequency table for negative sampling
        word_freq = {word: word_counts[word]**0.75 for word in vocab}
        total_freq = sum(word_freq.values())
        
        idx = 0
        for word in vocab:
            count = int((word_freq[word] / total_freq) * neg_table_size)
            self.neg_table[idx:idx+count] = list(self.wv.keys()).index(word) if word in self.wv else 0
            idx += count
        
        vocab_list = list(self.wv.keys())
        word2idx = {word: i for i, word in enumerate(vocab_list)}
        
        for epoch in range(self.epochs):
            total_loss = 0
            random.shuffle(sentences)
            for sentence in sentences:
                for i, word in enumerate(sentence):
                    if word not in self.wv:
                        continue
                    
                    center_idx = word2idx.get(word)
                    if center_idx is None:
                        continue
                    
                    context_start = max(0, i - self.window)
                    context_end = min(len(sentence), i + self.window + 1)
                    
                    for j in range(context_start, context_end):
                        if i == j:
                            continue
                        
                        context_word = sentence[j]
                        if context_word not in self.wv:
                            continue
                        
                        context_idx = word2idx.get(context_word)
                        if context_idx is None:
                            continue
                        
                        # Skip-gram with negative sampling
                        center_vec = self.wv[word]
                        context_vec = self.wv[context_word]
                        
                        # Positive sample
                        score = np.dot(center_vec, context_vec)
                        sigmoid = 1 / (1 + np.exp(-score))
                        
                        # Gradient
                        grad_center = (sigmoid - 1) * context_vec
                        grad_context = (sigmoid - 1) * center_vec
                        
                        # Negative samples
                        for _ in range(5):
                            neg_idx = self.neg_table[np.random.randint(len(self.neg_table))]
                            neg_word = vocab_list[neg_idx]
                            if neg_word == context_word:
                                continue
                            
                            neg_vec = self.wv[neg_word]
                            score_neg = np.dot(center_vec, neg_vec)
                            sigmoid_neg = 1 / (1 + np.exp(-score_neg))
                            
                            grad_center += sigmoid_neg * neg_vec
                            
                            # Update negative word vector
                            self.wv[neg_word] -= self.lr * sigmoid_neg * center_vec
                        
                        # Update center and context vectors
                        self.wv[word] -= self.lr * grad_center
                        self.wv[context_word] -= self.lr * grad_context
                        
                        total_loss += -np.log(sigmoid + 1e-10)
            
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss:.4f}")
    
    def save(self, path):
        torch.save(self.wv, path)
    
    @classmethod
    def load(cls, path):
        model = cls.__new__(cls)
        model.wv = torch.load(path)
        return model


class Preprocess():
    def __init__(self, sentences, sen_len, w2v_path="./w2v.model"):
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []
    
    def get_w2v_model(self):
        # load word to vector model
        self.wv_dict = torch.load(self.w2v_path)
        self.embedding_dim = len(list(self.wv_dict.values())[0])
    
    def add_embedding(self, word):
        # add word into embedding
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
        else:
            raise NotImplementedError

        for i, word in enumerate(self.wv_dict):
            print('get words #{}'.format(i+1), end='\r')
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(torch.tensor(self.wv_dict[word]))
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
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)
    
    def labels_to_tensor(self, y):
        # turn labels into tensors
        y = [int(label) for label in y]
        return torch.LongTensor(y)


class SenDataset(data.Dataset):
    """
    Expected data shape like:(data_num, data_len)
    Data can be a list of numpy array or a list of lists
    input data shape : (data_num, seq_len, feature_dim)

    __len__ will return the number of data
    """
    def __init__(self, X, y):
        self.data = X
        self.label = y
    def __getitem__(self, idx):
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]
    def __len__(self):
        return len(self.data)
