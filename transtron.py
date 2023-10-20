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
from encoder import Encoder
from decoder import Decoder
# warning表示off
import warnings
warnings.simplefilter('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Postnet
class Postnet(nn.Module):
    def __init__(
        self,
        in_dim=80,
        layers=5,
        channels=512,
        kernel_size=5,
        dropout=0.5,
    ):
        super().__init__()
        postnet = nn.ModuleList()
        for layer in range(layers):
            in_channels = in_dim if layer == 0 else channels
            out_channels = in_dim if layer == layers - 1 else channels
            postnet += [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            ]
            if layer != layers - 1:
                postnet += [nn.Tanh()]
            postnet += [nn.Dropout(dropout)]
        self.postnet = nn.Sequential(*postnet)

    def forward(self, xs):
        return self.postnet(xs)

# LengthRegulator で使う。        
def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad

# LengthRegulator エンコーダー出力を、各音素の duration でアップサンプリングし、decoder_targets を作る。
class LengthRegulator(torch.nn.Module):
    def __init__(self,pad_value = 0.0):
        super().__init__()
        self.pad_value = pad_value
        
    def forward(self, xs, ds ):
        #[print( "shape of x:{}, shape of d:{}".format( torch.tensor(x).size(), torch.tensor( d).size())) for x, d in zip(xs, ds)]
        # x = ( 1, 2 ) を d = (3, 5 )で ( 1,1,1,2,2,2,2,2 )にアップサンプリングする。
        repeat = [torch.repeat_interleave(x, d, dim=0) for x, d in zip(xs, ds)]
        return pad_list(repeat, self.pad_value)

# DurationPredictor で使う
class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x

#DuationPredictor エンコーダー出力の時間軸は各音素である。その各音素をどのくらいアップサンプリングするかを duration を学習し推論する。
class DurationPredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, input_size, filter_size, kernel_size, dp_dropout = 0.5 ):
        super(DurationPredictor, self).__init__()

        self.input_size = input_size
        self.filter_size = filter_size
        self.kernel = kernel_size
        self.conv_output_size = filter_size
        self.dropout = dp_dropout

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer1 = nn.Linear( self.conv_output_size, 1 )
        #self.offset = 1

    def forward(self, encoder_output, is_inference = False ):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer1( out )
        out = out.squeeze(-1)
        if is_inference == True:
            #out = torch.clamp(
            #    torch.round(out.exp() - self.offset), min=0
            #    ).long() # avoid negative value
            out = torch.clamp( torch.round( out ), min = 0 ).long()
        
        return out
        
class Transtron(nn.Module):
    def __init__(self,
            num_vocab=59,
            embed_dim=512,
            conv_layers=3,
            conv_channels=512,
            conv_kernel_size=5,
            enc_hidden_dim = 512,
            num_enc_layers = 6,
            enc_num_heads = 4,
            enc_dropout_rate = 0.1,
            conv_dropout_rate = 0.1,
            enc_input_maxlen = 300,
            enc_ffn_dim = 2048,
            enc_kernel_size = [5,1],
            enc_filter_size = 2048,
            decoder_hidden_dim=512,
            prenet_in_dim = 512,
            out_dim=80,
            num_dec_layers=6,
            prenet_layers=2,
            prenet_hidden_dim=512,
            prenet_dropout=0.5,
            dec_ffn_dim=2048,
            dec_dropout_rate = 0.1,
            dec_input_maxlen=3000,
            dec_num_heads = 4,
            dec_kernel_size = [5,1],
            dec_filter_size = 2048,
            postnet_in_dim=80,
            postnet_layers=5,
            postnet_channels=512,
            postnet_kernel_size=5,
            postnet_dropout=0.5,
            dp_input_size = 512,
            dp_filter_size = 512,
            dp_kernel_size = 3,
            dp_dropout = 0.4,
        ):
        super().__init__()
        self.encoder = Encoder(
            num_vocab,
            embed_dim,
            conv_layers,
            conv_channels,
            conv_kernel_size,
            enc_hidden_dim,
            num_enc_layers,
            enc_num_heads,
            enc_dropout_rate,
            conv_dropout_rate,
            enc_input_maxlen,
            enc_ffn_dim 
        )
        self.decoder = Decoder(
            decoder_hidden_dim,
            prenet_in_dim,
            out_dim,
            num_dec_layers,
            prenet_layers,
            prenet_hidden_dim,
            prenet_dropout,
            dec_ffn_dim,
            dec_dropout_rate,
            dec_input_maxlen,
            dec_num_heads,
            dec_kernel_size,
            dec_filter_size,
        )
        self.postnet = Postnet(
            postnet_in_dim,
            postnet_layers,
            postnet_channels,
            postnet_kernel_size,
            postnet_dropout
        )
        self.duration_prediction = DurationPredictor(
            dp_input_size, 
            dp_filter_size,
            dp_kernel_size,
            dp_dropout,
        )
        self.length_regulator = LengthRegulator()


    def forward(self, seq, in_lens, decoder_targets, out_lens, dur_feats):
        # エンコーダによるテキストに潜在する表現の獲得
        encoder_outs, att_ws_enc = self.encoder(seq, in_lens)
        
        # 学習時、npy ファイルから与えられた duration を元に、length_regurator で、エンコーダー in_feats の T から デコーダー out_feats の T へ。  
        decoder_targets1 = self.length_regulator( encoder_outs, dur_feats )
        
        # デコーダーの T の微調整。これをしないと、MSELoss の前の mask でエラーになる。
        decoder_targets2 = torch.zeros( decoder_targets.size(0), decoder_targets.size(1), encoder_outs.size(2) )

        if decoder_targets1.size(1) < decoder_targets2.size(1):
            decoder_targets2[:,:decoder_targets1.size(1),:] = decoder_targets1[:,:,:]
        else:
            decoder_targets2[:,:,:] = decoder_targets1[:,:decoder_targets2.size(1),:]
        
        # デコーダによるメルスペクトログラム の予測
        outs, att_ws_dec1, att_ws_dec2 = self.decoder( encoder_outs, decoder_targets2 )

        att_ws = {}
        att_ws["enc"] = att_ws_enc
        att_ws["dec1"] = att_ws_dec1
        att_ws["dec2"] = att_ws_dec2

        
        # Post-Net によるメルスペクトログラムの残差の予測
        outs_fine = outs + self.postnet(outs)

        # (B, C, T) -> (B, T, C)
        outs = outs.transpose(2, 1)
        outs_fine = outs_fine.transpose(2, 1)

        # 学習時に duration_prediction の学習のために計算している。
        duration_predict = self.duration_prediction( encoder_outs, is_inference =False )

        return outs, outs_fine, duration_predict, att_ws
    
    #推論関数
    @torch.no_grad()
    def inference(self, in_feats ):
    
        """Performs inference over one batch of inputs using greedy decoding."""
        in_feats = torch.unsqueeze( in_feats, axis = 0 )
        in_lens = []
        for feats in ( in_feats):
            in_lens.append( len( feats ))
        # エンコーダによるテキストに潜在する表現の獲得
        encoder_outs, att_ws_enc = self.encoder(in_feats, in_lens)

        # 推論時、duration の予測
        duration_predict = self.duration_prediction( encoder_outs, is_inference = True )        
        
        # エンコーダー in_feats の T から デコーダー out_feats の T へ。
        decoder_targets1 = self.length_regulator( encoder_outs, duration_predict )
    
        # デコーダーによる予測。
        outs, att_ws_dec1, att_ws_dec2 = self.decoder( encoder_outs, decoder_targets1) 

        # 残差の予測
        outs_fine = outs + self.postnet(outs)

        # (B, C, T) -> (B, T, C)
        outs = outs.transpose(2, 1)
        outs_fine = outs_fine.transpose(2, 1)
    
        att_ws = {}
        att_ws["enc"] = att_ws_enc
        att_ws["dec1"] = att_ws_dec1
        att_ws["dec2"] = att_ws_dec2
    
        return outs[0], outs_fine[0], att_ws
    
    #attention matrix の保存。
    def save_att_matrix( self, filename ):
        ''' Attention行列を画像にして保存する
        utt:      出力する、バッチ内の発話番号
        filename: 出力ファイル名
        '''
        # decoderのsave_att_matrixを実行
        self.decoder.save_att_matrix(filename)
        
'''        
def main():
      
    model = Transtron()

    a = torch.ones( ( 8, 100 ) ).long()
    b = torch.ones( ( 8 ) )
    c = torch.ones( ( 8, 1000, 80 ) )
    d = torch.ones( ( 8 ) )
    e = torch.ones( ( 8, 100 ) ).long()
    
    outs, outs_fine, duration_predict, att_ws = model( a, b, c, d, e  )
    
    model.save_att_matrix( './fig/test.png' )
    
    #aw = torch.stack( aw, dim = 0 )

    print( "size of outs:{}".format( outs.size() ))
    print( "size of outs_fine:{}".format( outs_fine.size() ))
    print( "size of duration predict:{}".format( duration_predict.size() ))

if __name__ == "__main__":
    main()
'''
