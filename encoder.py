import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import japanize_matplotlib
import os
from ttslearn.env import is_colab
from os.path import exists
from transformer import ResidualAttentionBlock
# warning表示off
import warnings
warnings.simplefilter('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(
        self,
        num_vocab=59,
        embed_dim=512,
        conv_layers=3,
        conv_channels=512,
        conv_kernel_size=5,
        enc_hidden_dim = 512,
        num_enc_layers = 6,
        num_heads = 4,
        enc_dropout_rate = 0.1,
        conv_dropout_rate = 0.1,
        input_maxlen = 300,
        ffn_dim = 2048,
        enc_kernel_size = [5,1],
        enc_filter_size = 2048
    ):
        super(Encoder, self).__init__()

        # 文字の埋め込み表現
        self.embed = nn.Embedding(num_vocab, embed_dim, padding_idx=0)

        # 1 次元畳み込みの重ね合わせ：局所的な時間依存関係のモデル化
        convs = nn.ModuleList()
        for layer in range(conv_layers):
            in_channels = embed_dim if layer == 0 else conv_channels
            out_channels = enc_hidden_dim if layer == conv_layers - 1 else conv_channels
            #print( " in_channels:{}".format( in_channels ))
            #print( " out_channels:{}".format( out_channels ))
            convs += [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    conv_kernel_size,
                    padding=(conv_kernel_size - 1) // 2,
                    bias=False,  # この bias は不要です
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(conv_dropout_rate),
            ]
        self.convs = nn.Sequential(*convs)

        # position_embedding
        self.pos_emb = nn.Embedding(input_maxlen, embed_dim).to(device)

        # Transformer Attention Block
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(embed_dim, num_heads, cross_attention = False, kernel_size = enc_kernel_size, n_ffn = enc_filter_size ) for _ in range(num_enc_layers)]
        )
        
        self.input_maxlen = input_maxlen
        
        self.dropout = nn.Dropout(p=enc_dropout_rate)
        
    def forward(self, x, in_lens ):
    
        # テキストベクトルの埋め込み
        emb = self.embed(x)
       
        # 1 次元畳み込みと embedding では、入力のサイズ が異なるので注意
        out = self.convs(emb.transpose(1, 2)).transpose(1, 2)
        
        # position embbeding
        maxlen = out.size()[1]
        positions = torch.range(start=0, end=self.input_maxlen - 1, step=1).to(torch.long).to(device)
        positions = self.pos_emb(positions.to(device))[:maxlen,:]
        x = out.to(device) + positions.to(device)
        x = self.dropout( x )
        
        # Transformer attention block
        attention_weights = []
        for i, block in enumerate( self.blocks ):
            x, attn1, attn2 = block(x, x, mask = None)
            attention_weights.append( attn1 )
            #attention_weights.append( attn2 )
            
        aw = torch.stack( attention_weights )
        
        return x, aw  # (batch_size, input_seq_len, d_model)
'''        
def main():
      
    encoder = Encoder()

    a = torch.ones( (32, 300 ) ).long()
    b = torch.ones( 32 )
    
    c, aw = encoder( a, b )
    
    #aw = torch.stack( aw, dim = 0 )

    print( "size of c:{}".format( c.size() ))
    print( "size of aw:{}".format( aw.size() ))


if __name__ == "__main__":
    main()
'''
