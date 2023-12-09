from model import SentimentLSTM, TransformerModel, Seq2SeqTransformer
import re
import torch
import torch.nn as nn
import tqdm
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
        return model

    def setup_loss(self):
        if self.cfg['option']['loss'] == 'BCE':
            loss = nn.BCEWithLogitsLoss()
        
        return loss


    def map_tokens_to_indices(self, x):
        x = re.sub(r"[^a-zA-Z0-9가-힣\s]", "", x)
        x = x.lower()
        tokens = word_tokenize(x)
        vocab = {word: idx + 1 for idx, word in enumerate(set(tokens))}
        indexed_tokens = [vocab[token] for token in tokens]

        return torch.tensor(indexed_tokens).unsqueeze(0)

    def training(self):
        model = self.model
        optimizer = self.optimizer
        train_loader = self.train_loader


        for i in range(self.cfg['solver']['epoch']):
            model.train()
            loss_sum = 0.0
            for i, data in enumerate(train_loader):
                input = self.map_tokens_to_indices(data['orginal_text'])
                target = self.map_tokens_to_indices(data['summary_text'])
                pred = model(input, target)
                

                print(pred)
