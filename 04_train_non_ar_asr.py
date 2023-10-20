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
from ttslearn.util import pad_1d, pad_2d
from pathlib import Path
from functools import partial
import librosa
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

# 学習のためのデータの場所。
in_paths_dev = sorted(Path("./dump/jsut_sr22050/norm/dev/in_non_ar_asr/").glob("*.npy"))
in_paths = sorted(Path("./dump/jsut_sr22050/norm/dev/in_non_ar_asr/").glob("*.npy"))
out_paths_dev = sorted(Path("./dump/jsut_sr22050/norm/dev/out_non_ar_asr/").glob("*.npy"))
out_paths = sorted(Path("./dump/jsut_sr22050/norm/dev/out_non_ar_asr/").glob("*.npy"))
dur_paths_dev = sorted(Path("./dump/jsut_sr22050/norm/dev/dur_non_ar_asr/").glob("*.npy"))
dur_paths = sorted(Path("./dump/jsut_sr22050/norm/dev/dur_non_ar_asr/").glob("*.npy"))

# モデル定義
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

#学習で必要な関数

def collate_fn_duration(batch):
    
    xs = [x[0] for x in batch]
    ys = [x[1] for x in batch]
    ds = [x[2] for x in batch]
    in_lens = [len(x) for x in xs]
    out_lens = [len(y) for y in ys]
    dur_lens = [len(d) for d in ds]
    in_max_len = max(in_lens)
    out_max_len = max(out_lens)
    dur_max_len = max(dur_lens)
    x_batch = torch.stack([torch.from_numpy(pad_1d(x, in_max_len)) for x in xs])
    y_batch = torch.stack([torch.from_numpy(pad_2d(y, out_max_len)) for y in ys])
    d_batch = torch.stack([torch.from_numpy(pad_1d(d, dur_max_len)) for d in ds])
    in_lens = torch.tensor(in_lens, dtype=torch.long)
    out_lens = torch.tensor(out_lens, dtype=torch.long)
    dur_lens = torch.tensor(dur_lens, dtype=torch.long)
    return x_batch, in_lens, y_batch, out_lens, d_batch, dur_lens
    

#class Dataset3(data_utils.Dataset):  # type: ignore
class Dataset3():  # type: ignore
    """Dataset for numpy files

    Args:
        in_paths (list): List of paths to input files
        out_paths (list): List of paths to output files
        dur_paths (list): List of paths to duration files
    """

    def __init__(self, in_paths, out_paths, dur_paths):
        self.in_paths = in_paths
        self.out_paths = out_paths
        self.dur_paths = dur_paths

    def __getitem__(self, idx):
        """Get a pair of input and target

        Args:
            idx (int): index of the pair

        Returns:
            tuple: input and target in numpy format
        """
        return np.load(self.in_paths[idx]), np.load(self.out_paths[idx]), np.load(self.dur_paths[idx])

    def __len__(self):
        """Returns the size of the dataset

        Returns:
            int: size of the dataset
        """
        return len(self.in_paths)

#dataset = Dataset3(in_paths, out_paths, dur_paths )
#dataset_dev = Dataset3(in_paths_dev, out_paths_dev, dur_paths_dev )
#collate_fn = partial(collate_fn_duration)
#data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, collate_fn=collate_fn, num_workers=0)
#data_loader_dev = torch.utils.data.DataLoader(dataset_dev, batch_size=8, collate_fn=collate_fn, num_workers=0)

#in_feats, in_lens, out_feats, out_lens, duration_target, dur_lens = next(iter(data_loader))
#print("入力特徴量のサイズ:", tuple(in_feats.shape))
#print("出力特徴量のサイズ:", tuple(out_feats.shape))
#print("duration_target のサイズ:", tuple(duration_target.shape))

def save_mel_graph( filename_mel, feats, out_feats_fine ):

    cmap = get_cmap()
    init_plot_style()
    #fig, ax = plt.subplots(2, 1, figsize=(8,6))
    fig, ax = plt.subplots(2, 1, figsize=(8,6))
    ax[0].set_title("Mel-spectrogram of natural speech")
    ax[1].set_title("Mel-spectrogram of Transtron output")
    #ax[2].set_title("error")

    sr = 22050

    mindb = min(feats.min(), out_feats_fine.min())
    #mindb = feats.min() 
    maxdb = max(feats.max(), out_feats_fine.max())
    #maxdb = feats.max()


    #hop_length = int(sr * 0.0125)
    #hop_length = int(sr * 0.0116100)
    hop_length = 256
    # 比較用に、自然音声から抽出された音響特徴量を読み込みむ
    mesh = librosa.display.specshow(
        feats.T, sr=sr, x_axis="time", y_axis="frames", hop_length=hop_length, cmap=cmap, ax=ax[0])
    mesh.set_clim(mindb, maxdb)
    fig.colorbar(mesh, ax=ax[0])
    mesh = librosa.display.specshow(
        out_feats_fine.T, sr=sr, x_axis="time", y_axis="frames", hop_length=hop_length, cmap=cmap, ax=ax[1])
    mesh.set_clim(mindb, maxdb)
    fig.colorbar(mesh, ax=ax[1])
   
    

    for i, a in enumerate( ax ):
        if i == len( ax ) - 1:
            a.set_xlabel("Time [sec]")
        a.set_ylabel("Mel filter channel")
    fig.tight_layout()

    # 図10-8
    savefig( filename_mel )


# 学習ログ解析
def save_hist_graph(history, history_val, filename):
    #損失と精度の確認
    #print(f'decoder_out_loss: {history[0,2]:.5f}, postnet_out_loss: {history[0,3]:.5f}, dur_loss: {history[0,4]:.5f}, loss: {history[0,5]:.5f}, lr: {history[0,6]:.5f}') 
    #print(f'decoder_out_loss: {history[-1,2]:.5f}, postnet_out_loss: {history[-1,3]:.5f}, dur_loss: {history[-1,4]:.5f}, loss: {history[-1,5]:.5f}, lr: {history[-1,6]:.5f}' )

    it_train = history[-1,0]
    #print( it_train )
    if it_train < 10:
      unit = 1
    else:
      unit = it_train // 10

    # 学習曲線の表示 (損失) train
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,2], 'y', label='decoder_out_loss')
    plt.plot(history[:,0], history[:,3], 'k', label='postnet_out_loss')
    plt.plot(history[:,0], history[:,4], 'r', label='duration_loss')
    plt.plot(history[:,0], history[:,5], 'b', label='loss')
    plt.xticks(np.arange(0,it_train+1, unit))
    plt.xlabel('繰り返し回数')
    plt.ylabel('損失')
    plt.title('学習曲線(損失)')
    plt.legend()
    #plt.show()

    filename_train = filename + "_train.png"
    plt.savefig(filename_train)
    plt.close()     

    it_val = history_val[-1,0]
    if it_val < 10:
      unit = 1
    else:
      unit = it_dval // 10    
    
    # 学習曲線の表示 (損失) val
    plt.figure(figsize=(9,8))
    plt.plot(history_val[:,0], history_val[:,2], 'y', label='decoder_out_loss')
    plt.plot(history_val[:,0], history_val[:,3], 'k', label='postnet_out_loss')
    plt.plot(history_val[:,0], history_val[:,4], 'r', label='duration_loss')
    plt.plot(history_val[:,0], history_val[:,5], 'b', label='loss')
    plt.xticks(np.arange(0,it_val+1, unit))
    plt.xlabel('繰り返し回数')
    plt.ylabel('損失')
    plt.title('学習曲線(損失)')
    plt.legend()
    #plt.show()

    filename_val = filename + "_val.png"
    plt.savefig(filename_val)
    plt.close()     

def main():

    #学習データの準備
    dataset = Dataset3(in_paths, out_paths, dur_paths )
    dataset_dev = Dataset3(in_paths_dev, out_paths_dev, dur_paths_dev )
    collate_fn = partial(collate_fn_duration)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, collate_fn=collate_fn, num_workers=0)
    data_loader_dev = torch.utils.data.DataLoader(dataset_dev, batch_size=8, collate_fn=collate_fn, num_workers=0)

    #途中経過保存のディレクトリ
    dir1 = Path("./fig/")
    dir2 = Path("./pt/" )
    dir3 = Path("./csv/" )

    dir1.mkdir(parents=True, exist_ok=True)
    dir2.mkdir(parents=True, exist_ok=True)
    dir3.mkdir(parents=True, exist_ok=True)

    # lr は学習率を表します
    optimizer = optim.Adam(model.parameters(), lr=0.0001 )

    # gamma は学習率の減衰係数を表します
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=500000)

    history = np.zeros((0, 7))
    history_dev = np.zeros((0, 6))
    cmap = get_cmap()
    init_plot_style()

    num_epochs = 700
    it_train = 0
    it_dev = 0

    for epoch in range( num_epochs ):
    
        model.train()
        total_decoder_out_loss = 0
        total_postnet_out_loss = 0
        total_dur_loss = 0
        total_loss = 0
        count = 0
        # DataLoader を用いたミニバッチの作成: ミニバッチ毎に処理する
        
        # 本当の計算で使う。
        phar = tqdm( range( len(data_loader) ), desc='train' )
        Iter_train = iter(data_loader)

        # テストジョブなど、短い時間で 1epoch 終えたいときに使う。
        #phar = tqdm( range( len(data_loader_dev) ), desc='train' )
        #Iter_train = iter(data_loader_dev)

        for i in phar:
            in_feats, in_lens, out_feats, out_lens, dur_feats, dur_lens = next(Iter_train)
            in_feats = in_feats.to(device)
            in_lens = in_lens.to(device)
            out_feats = out_feats.to(device)
            out_lens = out_lens.to(device)
            dur_feats1 = torch.round( dur_feats ).long().to(device)
            dur_feats2 = dur_feats.to(device)
            dur_lens = dur_lens.to(device)
            in_lens, indices = torch.sort(in_lens, dim=0, descending=True)
            in_feats, out_feats, out_lens = in_feats[indices], out_feats[indices], out_lens[indices]
            dur_feats1, dur_feats2, dur_lens = dur_feats1[indices], dur_feats2[indices], dur_lens[indices]
    
            count += 1
    
            # 順伝搬の計算
            outs, outs_fine, duration, _  = model(in_feats, in_lens, out_feats, out_lens, dur_feats1)
            duration = duration.to(device)
        
            # ゼロパディグした部分を損失関数のの計算から除外するためにマスクを適用します
            # Mask (B x T x 1)
            mask = make_non_pad_mask(out_lens).unsqueeze(-1).to(device)
            mask2 = make_non_pad_mask(in_lens).unsqueeze(-1).to(device)
            out_feats = out_feats.masked_select(mask)
            outs = outs.masked_select(mask)
            outs_fine = outs_fine.masked_select(mask)
            dur_feats2 = dur_feats2.masked_select(mask2.squeeze(-1))
            duration = duration.masked_select(mask2.squeeze(-1))
        
            # 損失の計算
            decoder_out_loss = nn.MSELoss()(outs, out_feats)
            postnet_out_loss = nn.MSELoss()(outs_fine, out_feats) 
            dur_loss = nn.MSELoss()(duration, dur_feats2 )
        
            # 損失の合計
            loss = decoder_out_loss + postnet_out_loss + dur_loss
        
            total_loss += loss.item()
            total_decoder_out_loss += decoder_out_loss.item()
            total_postnet_out_loss += postnet_out_loss.item()
            total_dur_loss += dur_loss.item()
        
            it_train += 1
            # optimizer に蓄積された勾配をリセット
            optimizer.zero_grad()
            # 誤差の逆伝播
            loss.backward()
            # パラメータの更新
            optimizer.step()
            # 学習率スケジューラの更新
            current_lr = optimizer.param_groups[0]["lr"]
            lr_scheduler.step()
        
            avg_loss = total_loss / count
        
            #プログレスバーに cer 表示
            phar.set_postfix( loss = avg_loss )   
        
        avg_loss = total_loss / count
        avg_decoder_out_loss = total_decoder_out_loss / count
        avg_postnet_out_loss = total_postnet_out_loss / count
        avg_dur_loss = total_dur_loss / count
    
        print(f"epoch: {epoch+1:3d}, train it: {it_train:6d}, decoder_out: {avg_decoder_out_loss :.5f}, postnet_out: {avg_postnet_out_loss :.5f}, duration: {avg_dur_loss :.5f}, loss: {avg_loss :.5f}")
        item = np.array([epoch+1, it_train, avg_decoder_out_loss , avg_postnet_out_loss , avg_dur_loss , avg_loss ,  current_lr ])
        history = np.vstack((history, item))
    
        # epoch 20 ごとに mel のグラフと attention_matrix のグラフを保存。訓練用。
        if ( epoch + 1 ) % 20 == 0:
        #if ( epoch + 1 ) % 1 == 0:
            Iter_train2 = iter(data_loader)
            in_feats, in_lens, out_feats, out_lens, dur_feats, dur_lens = next(Iter_train2)
            in_feats = in_feats.to(device)
            in_lens = in_lens.to(device)
            out_feats = out_feats.to(device)
            out_lens = out_lens.to(device)
            dur_feats1 = torch.round( dur_feats ).long().to(device)
            dur_lens = dur_lens.to(device)
            in_lens, indices = torch.sort(in_lens, dim=0, descending=True)
            in_feats, out_feats, out_lens = in_feats[indices], out_feats[indices], out_lens[indices]
            dur_feats1, dur_lens = dur_feats1[indices], dur_lens[indices]
            with torch.no_grad():
                outs, outs_fine, duration, _   = model(in_feats, in_lens, out_feats, out_lens, dur_feats1)
            feats = out_feats[0].cpu().detach().numpy()
            out_feats_fine = outs_fine[0].cpu().detach().numpy() 
    
            filename_mel = "./fig/mel_train_ep" + format(epoch+1,"04d")
            save_mel_graph( filename_mel, feats, out_feats_fine  )
            filename_att = "./fig/att_train_ep" + format(epoch+1,"04d")
            model.save_att_matrix( filename_att )

    
        model.eval()
        total_dev_decoder_out_loss = 0
        total_dev_postnet_out_loss = 0
        total_dev_dur_loss = 0
        total_dev_loss = 0
        count = 0

        # DataLoader を用いたミニバッチの作成: ミニバッチ毎に処理する
        phar = tqdm( range( len(data_loader_dev) ), desc='dev' )
        Iter_dev = iter(data_loader_dev)
    
        for i in phar:
            in_feats, in_lens, out_feats, out_lens, dur_feats, dur_lens = next(Iter_dev)
            in_feats = in_feats.to(device)
            in_lens = in_lens.to(device)
            out_feats = out_feats.to(device)
            out_lens = out_lens.to(device)
            dur_feats1 = torch.round( dur_feats ).long().to(device)
            dur_feats2 = dur_feats.to(device)
            dur_lens = dur_lens.to(device)
            in_lens, indices = torch.sort(in_lens, dim=0, descending=True)
            in_feats, out_feats, out_lens = in_feats[indices], out_feats[indices], out_lens[indices]
            dur_feats1, dur_feats2, dur_lens = dur_feats1[indices], dur_feats2[indices], dur_lens[indices]
        
            count += 1
   
            with torch.no_grad():
                outs, outs_fine,  duration, _ = model(in_feats, in_lens, out_feats, out_lens, dur_feats1)
            duration = duration.to(device)
            
            # ゼロパディグした部分を損失関数のの計算から除外するためにマスクを適用します
            # Mask (B x T x 1)
            mask = make_non_pad_mask(out_lens).unsqueeze(-1).to(device)
            mask2 = make_non_pad_mask(in_lens).unsqueeze(-1).to(device)
            out_feats = out_feats.masked_select(mask)
            outs = outs.masked_select(mask)
            outs_fine = outs_fine.masked_select(mask)
            dur_feats2 = dur_feats2.masked_select(mask2.squeeze(-1))
            duration = duration.masked_select(mask2.squeeze(-1))

        
            # 損失の計算
            dev_decoder_out_loss = nn.MSELoss()(outs, out_feats)
            dev_postnet_out_loss = nn.MSELoss()(outs_fine, out_feats) 
            dev_dur_loss = nn.MSELoss()(duration, dur_feats2 )

        
            # 損失の合計
            dev_loss = dev_decoder_out_loss + dev_postnet_out_loss + dev_dur_loss
        
            total_dev_loss += dev_loss.item()
            total_dev_decoder_out_loss += dev_decoder_out_loss.item()
            total_dev_postnet_out_loss += dev_postnet_out_loss.item()
            total_dev_dur_loss += dev_dur_loss.item()
        
            avg_dev_loss = total_dev_loss / count
        
            #プログレスバーに cer 表示
            phar.set_postfix( dev_loss = avg_dev_loss ) 

            it_dev += 1
        
        avg_dev_loss = total_dev_loss / count
        avg_dev_decoder_out_loss = total_dev_decoder_out_loss / count
        avg_dev_postnet_out_loss = total_dev_postnet_out_loss / count
        avg_dev_dur_loss = total_dev_dur_loss / count    
    
        print(f"epoch: {epoch+1:3d}, dev it: {it_dev:6d}, decoder_out: {avg_dev_decoder_out_loss:.5f}, postnet_out: {avg_dev_postnet_out_loss:.5f}, duration: {avg_dev_dur_loss:.5f}, loss: {avg_dev_loss:.5f}")
        item = np.array([epoch+1, it_dev, avg_dev_decoder_out_loss , avg_dev_postnet_out_loss , avg_dev_dur_loss , avg_dev_loss ])
        history_dev = np.vstack((history_dev, item))

        # epoch 20 ごとに mel のグラフと attention_matrix のグラフを保存。開発用。
        if ( epoch + 1 ) % 20 == 0:
        #if ( epoch + 1 ) % 1 == 0:
            Iter_dev2 = iter(data_loader_dev)
            in_feats, in_lens, out_feats, out_lens, dur_feats, dur_lens = next(Iter_dev2)
            in_feats = in_feats.to(device)
            in_lens = in_lens.to(device)
            out_feats = out_feats.to(device)
            out_lens = out_lens.to(device)
            dur_feats1 = torch.round( dur_feats ).long().to(device)
            #dur_feats2 = dur_feats.to(device)
            dur_lens = dur_lens.to(device)
            in_lens, indices = torch.sort(in_lens, dim=0, descending=True)
            in_feats, out_feats, out_lens = in_feats[indices], out_feats[indices], out_lens[indices]
            dur_feats1, dur_lens = dur_feats1[indices], dur_lens[indices]
            with torch.no_grad():
                outs, outs_fine, duration, _  = model(in_feats, in_lens, out_feats, out_lens, dur_feats1)
            feats = out_feats[0].cpu().detach().numpy()
            out_feats_fine = outs_fine[0].cpu().detach().numpy() 
    
            filename_mel = "./fig/mel_val_ep" + format(epoch+1,"04d")
            save_mel_graph( filename_mel, feats, out_feats_fine  )
            filename_att = "./fig/att_val_ep" + format(epoch+1,"04d")
            model.save_att_matrix( filename_att )
 
        #100エポックごとと最終に、pt とヒストリーの数値データを保存。
        if ( epoch + 1 ) == num_epochs:
            epoch_str = "final_" + format(epoch+1,"04d")
    
            hist_df = pd.DataFrame(history)
            filename_his = "./csv/history_" + epoch_str + ".csv"
            hist_df.to_csv(filename_his, header=False, index=False)
            hist_dev_df = pd.DataFrame(history_dev)
            filename_his_dev = "./csv/history_dev_" + epoch_str + ".csv"
            hist_dev_df.to_csv(filename_his_dev, header=False, index=False)    

            filename_hist_graph = "./fig/hist_graph_ep" + epoch_str
            save_hist_graph( history, history_dev, filename_hist_graph )
    
            save_path = "./pt/transtron_weight_training_" + epoch_str + ".pt"
            torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'dev_loss': avg_dev_loss,},
               save_path)
        elif (epoch + 1) % 100 == 0:
        #elif (epoch + 1) % 1 == 0:
            epoch_str = format(epoch+1,"04d")

            hist_df = pd.DataFrame(history)
            filename_his = "./csv/history_" + epoch_str + ".csv"
            hist_df.to_csv(filename_his, header=False, index=False)
            hist_dev_df = pd.DataFrame(history_dev)
            filename_his_dev = "./csv/history_dev_" + epoch_str + ".csv"
            hist_dev_df.to_csv(filename_his_dev, header=False, index=False)    

            filename_hist_graph = "./fig/hist_graph_ep" + epoch_str
            save_hist_graph( history, history_dev, filename_hist_graph )

            save_path = "./pt/transtron_weight_training_" + epoch_str + ".pt"
            torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'dev_loss': avg_dev_loss,},
               save_path)        


if __name__ == "__main__":
    main()

