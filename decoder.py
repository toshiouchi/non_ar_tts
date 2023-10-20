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

#
class Prenet(nn.Module):
    def __init__(self, in_dim, dec_hidden_dim, layers=2, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        prenet = nn.ModuleList()
        for layer in range(layers):
            in_dims = in_dim if layer == 0 else hidden_dim
            out_dims = dec_hidden_dim if layer == layers - 1 else hidden_dim
            prenet += [
                nn.Linear(in_dims, out_dims ).to(device),
                nn.ReLU().to(device),
                nn.Dropout(dropout).to(device) # added by Toshio Uchiyama
            ]
        self.prenet = nn.Sequential(*prenet)   
        self.prenet = nn.Sequential(*prenet)

    def forward(self, x):
        for layer in self.prenet:
            x = layer(x.to(device))
        return x
        
class Decoder(nn.Module):
    def __init__(
        self,
        decoder_hidden_dim=512,
        prenet_in_dim = 512,
        out_dim=80,
        layers=6,
        prenet_layers=2,
        prenet_hidden_dim=512,
        prenet_dropout=0.5,
        ffn_dim=2048,
        dropout_rate = 0.1,
        dec_input_maxlen=3000,
        num_heads = 4,
        dec_kernel_size = [5,1],
        dec_filter_size = 2048
    ):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads

        # Prenet
        self.prenet = Prenet(prenet_in_dim, decoder_hidden_dim,  prenet_layers, prenet_hidden_dim, dropout = prenet_dropout).to(device)

        # position embedding
        self.pos_emb = nn.Embedding(dec_input_maxlen, decoder_hidden_dim).to(device)

        #  Transformer Attention  Block
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(decoder_hidden_dim, num_heads, cross_attention=True, kernel_size = dec_kernel_size, n_ffn = dec_filter_size  ) for _ in range(layers)]
        )

        # 出力への projection 層
        proj_in_dim = decoder_hidden_dim
        self.feat_out = nn.Linear(proj_in_dim, out_dim, bias=False)
        
        self.dec_input_maxlen = dec_input_maxlen
        self.aw2 = None

    def forward(self, encoder_outs, decoder_targets=None):

        # Pre-Net
        prenet_out = self.prenet(decoder_targets)

        # position embedding
        maxlen = prenet_out.size()[1]
        positions = torch.range(start=0, end=self.dec_input_maxlen - 1, step=1).to(torch.long)
        positions = self.pos_emb(positions.to(device))[:maxlen,:]
        x = prenet_out + positions
        
        # Transformer attention block
        attention_weights1 = []
        attention_weights2 = []
        for i, block in enumerate( self.blocks ):
            x, attn1, attn2 = block(x, encoder_outs, mask=None)
            attention_weights1.append( attn1 )
            attention_weights2.append( attn2 )
            
        aw1 = torch.stack( attention_weights1 )
        aw2 = torch.stack( attention_weights2 )
        aw3 = torch.sum( aw2, dim = 2 )
        aw3 = torch.sum( aw3, dim = 0 )
        self.aw3 = aw3
        
        # attention の　hidden_dim から mel の　80 へ。
        outs = self.feat_out(x)
        outs = torch.permute(outs, (0, 2, 1))
        
        return outs, aw1, aw2
    
    # attention matrix のグラフを保存。
    #def save_att_matrix(self, utt, filename):
    def save_att_matrix(self, filename):
        '''
        Attention行列を画像にして保存する
        utt:      出力する、バッチ内の発話番号
        filename: 出力ファイル名
        '''
        att_mat = self.aw3[0].cpu().detach().numpy()

        # プロットの描画領域を作成
        plt.figure(figsize=(5,5))
        # カラーマップのレンジを調整
        att_mat -= np.max(att_mat)
        vmax = np.abs(np.min(att_mat)) * 0.0
        vmin = - np.abs(np.min(att_mat)) * 0.7
        # プロット
        plt.imshow(att_mat, 
                   cmap = 'gray',
                   vmax = vmax,
                   vmin = vmin,
                   aspect = 'auto')
        # 横軸と縦軸のラベルを定義
        plt.xlabel('Encoder index')
        plt.ylabel('Decoder index')

        # プロットを保存する
        plt.savefig(filename)
        plt.close()        
'''        
def main():
      
    decoder = Decoder()

    a = torch.ones( (8, 100, 512 ) )
    b = torch.ones( (8, 1000, 512 ) )
    
    c, aw1, aw2 = decoder( a, b  )
    
    decoder.save_att_matrix( './fig/test.png' )
    
    #aw = torch.stack( aw, dim = 0 )

    print( "size of c:{}".format( c.size() ))
    print( "size of aw2:{}".format( aw2.size() ))


if __name__ == "__main__":
    main()
'''
