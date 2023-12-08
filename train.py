from engine import Trainer
from config import get_config_dict



if __name__=='__main__':
    config = get_config_dict()
    trainer = Trainer(config)
    trainer.training()
