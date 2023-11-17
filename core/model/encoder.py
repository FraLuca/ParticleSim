
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

        layer_list = []

        for i in range(cfg.MODEL.ENCODER.NUM_LAYERS):
            if i == 0:
                layer_list.append(nn.Linear(cfg.MODEL.ENCODER.INPUT_SIZE, cfg.MODEL.ENCODER.HIDDEN_SIZE))
            elif i == (cfg.MODEL.ENCODER.NUM_LAYERS - 1):
                layer_list.append(nn.Linear(cfg.MODEL.ENCODER.HIDDEN_SIZE, cfg.MODEL.ENCODER.OUTPUT_SIZE))
            else:
                layer_list.append(nn.Linear(cfg.MODEL.ENCODER.HIDDEN_SIZE, cfg.MODEL.ENCODER.HIDDEN_SIZE))
            
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Dropout(cfg.MODEL.ENCODER.DROPOUT))

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)
    

class TransformerEncoder(nn.Module):
    '''
        The main point of the TF is that we treat the input as a sequence of 1D vectors.
        While the MLP treats it as a element with 47 channels, the tf treats it as a sequence of 47 elements with 1 channel.
    '''
    def __init__(self, cfg):
        super(TransformerEncoder, self).__init__()
        TF_EMBED_SIZE = 32
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=TF_EMBED_SIZE, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.output_layer = nn.Linear(TF_EMBED_SIZE, 1)
        self.dropout = nn.Dropout(cfg.MODEL.ENCODER.DROPOUT)
        self.activation = nn.ReLU()

        self.input_layer = nn.Linear(1, TF_EMBED_SIZE)
        self.pos_embed_layer = nn.Linear(3, TF_EMBED_SIZE)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, TF_EMBED_SIZE))
        # TODO: try with a learnable pos embedding
        self.pos_embedding = nn.Parameter(torch.Tensor([[0,3,0.5],[0,3,1.5],[0,3,2.5],[0,3,3.5],[0,3,4.5],[0,3,5.5],
                                           [1,0,0],[1,6,6],[2,0,6],[2,6,0],
                                           [3,0,0],[3,6,6],[4,0,6],[4,6,0],
                                           [5,0,0],[5,6,6],[6,0,6],[6,6,0],
                                           [7,0,0],[7,6,6],[8,0,6],[8,6,0],
                                           [9,0,0],[9,6,6],[10,0,6],[10,6,0],
                                           [11,0,0],[11,6,6],[12,0,6],[12,6,0],
                                           [13,0,0],[13,6,6],[14,0,6],[14,6,0],
                                           [15,0,0],[15,6,6],[16,0,6],[16,6,0],
                                           [17,0.5,0.5], [17,3,0.5], [17,5.5,0.5], [17,0.5,3], [17,3,3], [17,5.5,3], [17,0.5,5.5], [17,3,5.5], [17,5.5,5.5]
                                           ]), requires_grad=False)

        self.mlp_output = MLP(cfg)
    
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
        out = self.activation(self.output_layer(x[:,1:,:]))
        out = self.dropout(out)

        out = self.mlp_output(out.squeeze(-1))
        # if removed we return the CLS token x = x[:,0,:]
        # out = x[:,0,:]

        return out