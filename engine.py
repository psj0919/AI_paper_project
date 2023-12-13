from model import SentimentLSTM, TransformerModel, Seq2SeqTransformer
from transformers  import BertTokenizer, BertForSequenceClassification
import re
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import tqdm
from PyKomoran import *
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy

class Trainer():
    def __init__(self,cfg):
        self.cfg = cfg
        #self.device = self.setup_device()
        self.model = self.setup_model()
        self.optimizer = self.setup_optimizer()
        # self.scheduler = self.setup_scheduler()
        self.global_step = 0
        self.train_loader = self.get_dataloader()
        self.loss = self.setup_loss()

    def setup_optimizer(self):
        if self.cfg['option']['optim'] == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters())

        return optimizer

    def get_dataloader(self):
        from dataset import paperdata
        data  = paperdata(self.cfg['dataset']['data_path'])
        return data

    def setup_model(self):
        if self.cfg['model']['name'] == 'LSTM':
            model = SentimentLSTM(self.cfg['dataset']['max_size'], self.cfg['dataset']['embedding_dim'],self.cfg['dataset']['hidden_dim'],self.cfg['dataset']['output_dim'] ,self.cfg['dataset']['n_layers'] ,self.cfg['dataset']['bidirectional'] ,self.cfg['dataset']['dropout'])
        elif self.cfg['model']['name'] == 'transformer':
            model = TransformerModel(self.cfg['dataset']['max_size'], embedding_dim=256, hidden_dim=512, nhead=4, num_encoder_layers=2, num_classes=1,dropout=0.5)
        elif self.cfg['model']['name'] == 'seq2seq':
            model = Seq2SeqTransformer(self.cfg['dataset']['max_size'], embedding_dim= 256, hidden_dim = 512, nhead = 4, num_layers = 2)
        elif self.cfg['model']['name'] == 'bert':
            model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        return model

    def setup_loss(self):
        if self.cfg['option']['loss'] == 'BCE':
            loss = nn.BCEWithLogitsLoss()
        
        return loss


    def map_tokens_to_indices(self, x):
        tokens = word_tokenize(x)
        vocab = {word: idx + 1 for idx, word in enumerate(set(tokens))}
        indexed_tokens = [vocab[token] for token in tokens]

        return torch.tensor(indexed_tokens).unsqueeze(0), vocab

    def summarize_text(self, text):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        max_length = 512
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        summary_prob = probs[:, 0].item()
        predict_class = torch.argmax(logits, dim=1).item()
        summary_text = tokenizer.decode(inputs["input_ids"][0])

        return summarize_text


    def training(self):
        model = self.model
        optimizer = self.optimizer
        train_loader = self.train_loader
        komoran = Komoran("EXP")

        for i in range(self.cfg['solver']['epoch']):
            model.train()
            loss_sum = 0.0
            for i, data in enumerate(train_loader):
                input = komoran.get_plain_text(data['orginal_text'])
                target = komoran.get_plain_text(data['summary_text'])
                input, input_vocab = self.map_tokens_to_indices(input)
                target, target_vocab = self.map_tokens_to_indices(data['summary_text'])
                pred = model(input, target)
                probe = softmax(pred, dim = -1)
                max_prob_index = torch.argmax(probe, dim=-1)
                decode_word = input[max_prob_index.item()]
            
