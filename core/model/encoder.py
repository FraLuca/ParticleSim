
import torch
import torch.nn as nn

def build_encoder(cfg):
    if cfg.MODEL.ENCODER.NAME == 'mlp':
        return MLP(cfg)
    elif cfg.MODEL.ENCODER.NAME == 'tf':
        return TransformerEncoder(cfg)
    else:
        raise NotImplementedError('Only MLP encoder is implemented')
    


def load_encoder(cfg):
    encoder = build_encoder(cfg)
    encoder.load_state_dict(torch.load(cfg.ENCODER_PATH)['state_dict'])
    return encoder
    

class MLP(nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(cfg.MODEL.ENCODER.INPUT_SIZE, cfg.MODEL.ENCODER.HIDDEN_SIZE)
        self.fc2 = nn.Linear(cfg.MODEL.ENCODER.HIDDEN_SIZE, cfg.MODEL.ENCODER.HIDDEN_SIZE)
        self.fc3 = nn.Linear(cfg.MODEL.ENCODER.HIDDEN_SIZE, cfg.MODEL.ENCODER.HIDDEN_SIZE)
        self.fc4 = nn.Linear(cfg.MODEL.ENCODER.HIDDEN_SIZE, cfg.MODEL.ENCODER.HIDDEN_SIZE)
        self.fc5 = nn.Linear(cfg.MODEL.ENCODER.HIDDEN_SIZE, cfg.MODEL.ENCODER.HIDDEN_SIZE)
        self.fc6 = nn.Linear(cfg.MODEL.ENCODER.HIDDEN_SIZE, cfg.MODEL.ENCODER.OUTPUT_SIZE)
        self.dropout = nn.Dropout(cfg.MODEL.ENCODER.DROPOUT)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.dropout(x)
        x = self.activation(self.fc4(x))
        x = self.dropout(x)
        x = self.activation(self.fc5(x))
        x = self.dropout(x)
        x = self.fc6(x)
        return x

class TransformerEncoder(nn.Module):
    '''
        The main point of the TF is that we treat the input as a sequence of 1D vectors.
        While the MLP treats it as a element with 47 channels, the tf treats it as a sequence of 47 elements with 1 channel.
    '''
    def __init__(self, cfg):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.MODEL.ENCODER.HIDDEN_SIZE, nhead=cfg.MODEL.ENCODER.NHEAD)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=cfg.MODEL.ENCODER.NUM_LAYERS)
        self.fc = nn.Linear(cfg.MODEL.ENCODER.INPUT_SIZE, cfg.MODEL.ENCODER.OUTPUT_SIZE)
        self.dropout = nn.Dropout(cfg.MODEL.ENCODER.DROPOUT)
        self.activation = nn.ReLU()

        self.input_layer = nn.Linear(cfg.MODEL.ENCODER.INPUT_SIZE, cfg.MODEL.ENCODER.HIDDEN_SIZE)
        self.pos_embed_layer = nn.Linear(3, cfg.MODEL.ENCODER.HIDDEN_SIZE)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.MODEL.ENCODER.HIDDEN_SIZE))
        # TODO: try with a learnable pos embedding
        self.pos_embedding = torch.Tensor([[0,3,0.5],[0,3,1.5],[0,3,2.5],[0,3,3.5],[0,3,4.5],[0,3,5.5],
                                           [1,0,0],[1,6,6],[2,0,6],[2,6,0],
                                           [3,0,0],[3,6,6],[4,0,6],[4,6,0],
                                           [5,0,0],[5,6,6],[6,0,6],[6,6,0],
                                           [7,0,0],[7,6,6],[8,0,6],[8,6,0],
                                           [9,0,0],[9,6,6],[10,0,6],[10,6,0],
                                           [11,0,0],[11,6,6],[12,0,6],[12,6,0],
                                           [13,0,0],[13,6,6],[14,0,6],[14,6,0],
                                           [15,0,0],[15,6,6],[16,0,6],[16,6,0],
                                           [17,0.5,0.5], [17,3,0.5], [17,5.5,0.5], [17,0.5,3], [17,3,3], [17,5.5,3], [17,0.5,5.5], [17,3,5.5], [17,5.5,5.5]
                                           ]).unsqueeze(0)
    
    def forward(self, x):
        B, S = x.shape

        # input embedding
        x = self.input_layer(x.unsqueeze(-1))

        # Add CLS token and positional encoding
        x = x + self.pos_embed_layer(self.pos_embedding)
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        
        x = self.transformer_encoder(x)

        # can be removed?
        x = self.activation(self.fc(x[:,0,:]))
        x = self.dropout(x)
        # if removed we return the CLS token x = x[:,0,:]

        return x