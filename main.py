import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class Trainer():
        def __init__(self, cfg,device=torch.device('cpu')):
            self.cfg = cfg
            self.device = device
            self.model = self.setup_model(self.device).to(self.device)



        def setup_model(self):













