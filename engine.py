from model import LSTMModel
import torch
import torch.nn as nn
import tqdm

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
            model = LSTMModel(self.cfg['dataset']['max_size'], self.cfg['dataset']['embedding_dim'],self.cfg['dataset']['hidden_dim'],self.cfg['dataset']['output_dim'] ,self.cfg['dataset']['n_layers'] ,self.cfg['dataset']['bidirectional'] ,self.cfg['dataset']['dropout'])
        
        return model

    def setup_loss(self):
        if self.cfg['option']['loss'] == 'BCE':
            loss = nn.BCEWithLogitsLoss()
        
        return loss
    def training(self):
        model = self.model
        optimizer = self.optimizer
        train_loader = self.train_loader



        for i in range(self.cfg['solver']['epoch']):
            model.train()
            loss_sum = 0.0
            pbar = tqdm(enumerate(train_loader), len(train_loader))
            for curr_step, data in pbar:
                output = model(data)


    
