import sys
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
from torch import optim
from ttslearn.notebook import get_cmap, init_plot_style, savefig
from ttslearn.util import make_non_pad_mask
from ttslearn.tacotron import Tacotron2TTS
from tqdm import tqdm
from IPython.display import Audio
import pandas as pd
from pathlib import Path
#学習で必要なミニバッチデータ
from pathlib import Path
from functools import partial
import librosa
import pyopenjtalk
# この実装は後述します
from ttslearn.tacotron.frontend.openjtalk import pp_symbols
from ttslearn.tacotron.frontend.openjtalk import text_to_sequence, pp_symbols
from ttslearn.util import find_lab, find_feats
from ttslearn.dsp import logmelspectrogram_to_audio
import subprocess
import datetime
from transtron import Transtron
# warning表示off
import warnings
warnings.simplefilter('ignore')
# デフォルトフォントサイズ変更
plt.rcParams['font.size'] = 14
# デフォルトグラフサイズ変更
plt.rcParams['figure.figsize'] = (6,6)
# デフォルトで方眼表示ON
plt.rcParams['axes.grid'] = True
# numpyの表示桁数設定
np.set_printoptions(suppress=True, precision=5)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device( "cpu" )


model = Transtron(
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
    enc_kernel_size = 3,
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
    dec_kernel_size = 3,
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
)

model = model.to(device)

model.eval()

#推論のための関数
@torch.no_grad()
def inference(in_feats ):
    
    """Performs inference over one batch of inputs using greedy decoding."""
    in_feats = torch.unsqueeze( in_feats, axis = 0 )
    in_lens = []
    for feats in ( in_feats):
        in_lens.append( len( feats ))

    # エンコーダによるテキストに潜在する表現の獲得
    #model = model.to(torch.device("cpu" ))
    encoder_outs, att_ws_enc = model.encoder(in_feats, torch.tensor(in_lens))


    duration_predict = model.duration_prediction( encoder_outs, is_inference = True )
    duration_predict = torch.round( duration_predict ).long()
    
    decoder_targets1 = model.length_regulator( encoder_outs, duration_predict )    
    
    outs, att_ws_dec1, att_ws_dec2 = model.decoder(encoder_outs, decoder_targets1) 

    outs_fine = outs + model.postnet(outs)

    # (B, C, T) -> (B, T, C)
    outs = outs.transpose(2, 1)
    outs_fine = outs_fine.transpose(2, 1)
    
    att_ws = {}
    att_ws["enc"] = att_ws_enc
    att_ws["dec1"] = att_ws_dec1
    att_ws["dec2"] = att_ws_dec2
    
    return outs[0], outs_fine[0], att_ws  

def main(text, model):

    labels = pyopenjtalk.extract_fullcontext(text)
    
    PP = pp_symbols(labels)

    phone_to_id = {'~': 0, '^': 1, 'm': 2, 'i[': 3, 'z': 4, 'u': 5, 'o#': 6, 'a[': 7, 'r': 8, 'e]': 9,\
               'e': 10, 'sh': 11, 'i': 12, 'a': 13, 'k': 14, 'a#': 15, 'w': 16, 'n': 17, 'a]': 18, 't': 19,\
               'o': 20, 'd': 21, 's': 22, '$': 23, 'o[': 24, 'y': 25, 'o]': 26, 'b': 27, '_': 28, 'e[': 29,\
               'N': 30, 'u[': 31, 'ry': 32, 'j': 33, 'g': 34, 'i]': 35, 'h': 36, 'ts': 37, 'cl': 38, 'u]': 39,\
               'ny': 40, 'i#': 41, 'p': 42, 'e#': 43, 'f': 44, 'gy': 45, 'ky': 46, 'ch': 47, 'N#': 48, 'u#': 49,\
               '?': 50, 'hy': 51, 'my': 52, 'N]': 53, 'by': 54, 'py': 55, 'cl[': 56, 'v': 57, 'dy': 58}
    onso = ["m","i","z","u","o","a","r","e","sh","k","w","n","U","t","d","s","y","b","_","N","ry",\
        "I","j","g","h","ts","cl","ny","p","f","gy","ky","ch","hy","my","by","py","v","dy", "$", "^", "?"]

    n = 0
    Together_Prosody = []
    for PPP in PP:
        if PPP in onso:
            Together_Prosody.append( PPP )
            n += 1
        else:
            Together_Prosody[n-1] = Together_Prosody[n-1] + PPP

    in_feats = []
    for Together in Together_Prosody:
        in_feats.append( phone_to_id[ Together ] )

    in_feats_np = np.array( in_feats )


    in_feats = torch.tensor(in_feats, dtype=torch.long)

    with torch.no_grad():
        out_feats, out_feats_fine, alignment0 = inference( in_feats )

    dt_now = str( datetime.datetime.now() ).replace( " ", "-" ).replace( ".", "-" ).replace( ":", "-" )
    filename0 = "test" + dt_now + ".npy"
    filename = "test" + dt_now + "_generated_e2e.wav"

    np.save( "./hifi-gan-master/test_mel_files/" + filename0 , out_feats_fine.cpu().data.numpy().T)
    
    text2 = "python3 hifi-gan-master/inference_e2e.py --input_mels_dir hifi-gan-master/test_mel_files --output_dir hifi-gan-master/generated_files_from_mel --checkpoint_file hifi-gan-master/cp_hifigan/g_00087000"
    #text2 = "hifi-gan-master/05_inf.sh"
    subprocess.run(text2, shell=True)

    filename3 = "hifi-gan-master\\test_mel_files\\" + filename0
    #text3 = "del hifi-gan-master\\test_mel_files\\" + filename0
    #subprocess.run( text3, shell=True)
    
    return filename3

if __name__ == "__main__":

    args = sys.argv

    save_path2 = "./pt/" + args[1]
    checkpoint = torch.load(save_path2, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    text = "本日は晴天なり。"

    filename = main(text, model)
    
    print( "inferenced voice file:{}".format( filename ) )
