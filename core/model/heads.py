import torch.nn as nn

def build_classifier(cfg):
    return MLP_CLS(cfg)

def build_regressor(cfg):
    return MLP_REG(cfg)


class MLP_CLS(nn.Module):
    def __init__(self, cfg):
        super(MLP_CLS, self).__init__()
        self.fc1 = nn.Linear(cfg.MODEL.ENCODER.HIDDEN_SIZE, cfg.MODEL.CLASSIFIER.HIDDEN_SIZE)
        self.fc3 = nn.Linear(cfg.MODEL.CLASSIFIER.HIDDEN_SIZE, cfg.MODEL.CLASSIFIER.OUTPUT_SIZE)
        self.dropout = nn.Dropout(cfg.MODEL.CLASSIFIER.DROPOUT)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MLP_REG(nn.Module):
    def __init__(self, cfg):
        super(MLP_REG, self).__init__()
        self.fc1 = nn.Linear(cfg.MODEL.ENCODER.HIDDEN_SIZE, cfg.MODEL.REGRESSOR.HIDDEN_SIZE)
        self.fc3 = nn.Linear(cfg.MODEL.REGRESSOR.HIDDEN_SIZE, cfg.MODEL.REGRESSOR.OUTPUT_SIZE)
        self.dropout = nn.Dropout(cfg.MODEL.REGRESSOR.DROPOUT)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
