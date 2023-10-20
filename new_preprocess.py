import argparse
import sys
import os
sys.path.append( "./hifi-gan-master/" )
from meldataset import mel_spectrogram
#from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import torch
from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin as fe
from scipy.io import wavfile
from scipy.io.wavfile import read, write
from tqdm import tqdm
from ttslearn.tacotron.frontend.openjtalk import pp_symbols, text_to_sequence
from ttslearn.util import pad_1d

def get_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess for Tacotron",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("utt_list", type=str, help="utternace list")
    parser.add_argument("wav_root", type=str, help="wav root")
    parser.add_argument("lab_root", type=str, help="lab_root")
    parser.add_argument("out_dir", type=str, help="out directory")
    parser.add_argument("hi_root", type=str, help="hifi-gan-master root" )
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Sample rate")
    parser.add_argument("--n_fft", type=int, default=1024, help="n_fft")
    parser.add_argument("--n_mels", type=int, default=80, help="n_mels" )
    parser.add_argument("--hop_length", type=int, default=256, help="hop_length" )
    parser.add_argument("--win_length", type=int, default=1024, help="win_length" )
    parser.add_argument("--f_min", type=int, default=0, help="f_min" )
    parser.add_argument("--f_max", type=int, default=8000, help="f_max" )
    return parser

def get_mel(x, n_fft, n_mels, sample_rate, hop_length, win_length, f_min, f_max):
    return mel_spectrogram(x, n_fft, n_mels, sample_rate, hop_length, win_length, f_min, f_max)
    
def preprocess(
    wav_file,
    lab_file,
    sr,
    n_fft,
    n_mels,
    win_length,
    hop_length,
    f_min,
    f_max,
    in_dir,
    out_dir,
    dur_dir,
    hi_mels_dir,
    hi_wavs_dir
):
    assert wav_file.stem == lab_file.stem
    labels = hts.load(lab_file)
    # 韻律記号付き音素列の抽出
    PP = pp_symbols(labels.contexts)
    in_feats = text_to_sequence( PP )
    # 継続長モデルから duration を読み込み
    durations_orig = fe.duration_features( labels )
    
    # メルスペクトログラムの計算
    _sr, x = wavfile.read(wav_file)
    # x は、melspectrogram を計算するために。xx は、wav ファイルに書き込むために sr=22050Hz の PCM16。
    xx = x
    xx = librosa.resample( xx.astype(np.float32), orig_sr=_sr, target_sr=sr).astype(np.int16)

    if x.dtype in [np.int16, np.int32]:
        x = (x / np.iinfo(x.dtype).max).astype(np.float64)
    x = librosa.resample(x, orig_sr=_sr, target_sr=sr)
    x = x[ np.newaxis, :]
    x = torch.from_numpy(x).float()
    out_feats = get_mel( x, n_fft, n_mels, sr, hop_length, win_length, f_min, f_max )
    out_feats = torch.squeeze( out_feats, dim = 0 ).cpu().detach().numpy()
    
    # 冒頭と末尾の非音声区間の長さを調整
    assert "sil" in labels.contexts[0] and "sil" in labels.contexts[-1]
    start_frame = int(labels.start_times[1] / 116100)
    end_frame = int(labels.end_times[-2] / 116100)


    # 冒頭： 50 ミリ秒、末尾： 100 ミリ秒 22050Hz の hop_length (256frame) は、0.0116100 秒。
    start_frame = max(0, start_frame - int(0.050 / 0.01161))
    end_frame = min(out_feats.shape[1], end_frame + int(0.100 / 0.01161))

    out_feats = out_feats[:,start_frame:end_frame]   # HiFiGan 用 melspectrogram
    out_feats2 = np.transpose( out_feats, ( 1, 0 ))  # Transtron 用 melspectrogram
    
    # duration の計算。
    # 基本音素 + "^" BOS + "$" EOS。 duration_orig の継続長に対応した音素と記号。韻律は考えない。 
    onso = ["m","i","z","u","o","a","r","e","sh","k","w","n","U","t","d","s","y","b","_","N","ry",\
        "I","j","g","h","ts","cl","ny","p","f","gy","ky","ch","hy","my","by","py","v","dy", "$", "^"]
        
    n = 0
    duration_PP = []
    for PPP in PP:
        if PPP in onso:
            duration_PP.append( float( durations_orig[n] ) )
            n += 1
        else:
            duration_PP.append( 0 )

    durations_PP_np = np.array( duration_PP )

    durations_PP_np[0] = 10 # BOS ^ については、0.05秒  0.3秒が 60 なので、
    durations_PP_np[-1] = 20 # EOS $ については、0.10秒
    
    out_len = out_feats.shape[1]
    
    # 継続長モデル（16000Hz)の duration の sum を、全体で、out_feats.shape[1] になるように規格化
    durations = durations_PP_np * out_len / np.sum( durations_PP_np )
    #durations = np.round( durations.astype(np.float) ).astype(np.long)
    
    # 時間領域で音声の長さを調整
    xx = xx[int(start_frame * 256) :]
    length = 256 * out_feats.shape[1]
    xx = pad_1d(xx, length) if len(xx) < length else xx[:length]
    
    # 特徴量のアップサンプリングを行う都合上、音声波形の長さはフレームシフト(hop_length)で割り切れる必要があります
    assert len(xx) % 256 == 0
    
    # save to files
    
    in_feats_np = np.array( in_feats )
    #print( "shape of in_feats:{}".format( in_feats_np.shape))
    #print( "shape of durations:{}".format( durations.shape ))
    #print( "shape of out_feats:{}".format( out_feats.shape ))
    #print( "shape of out_feats2:{}".format( out_feats2.shape ))
    #print( "shape of xx:{}".format( xx.shape ))
    
    utt_id = lab_file.stem
    np.save(in_dir / f"{utt_id}-feats.npy", in_feats, allow_pickle=False)

    in_dir2 = str(in_dir).replace( "org", "norm" )
    np.save(in_dir2 + "/" +  f"{utt_id}-feats.npy", in_feats, allow_pickle=False)
    
    np.save(
        out_dir /  f"{utt_id}-feats.npy",
        out_feats2.astype(np.float32),
        allow_pickle=False,
    )

    #out_dir2 = str(out_dir).replace( "org", "norm" )
    #np.save(
    #    out_dir2 + "/" +  f"{utt_id}-feats.npy",
    #    out_feats2.astype(np.float32),
    #    allow_pickle=False,
    #)

    #out_dir_dur = str( out_dir2 ).replace( "out_tacotron", "out_duration" )
    np.save(
        dur_dir /  f"{utt_id}-feats.npy",
        #durations.astype(np.long),
        durations.astype(np.float32),
        allow_pickle=False,
    )

    np.save(
        hi_mels_dir / f"{utt_id}-feats.npy",
        out_feats.astype(np.float32),
        allow_pickle=False,
    )
    writefilename = hi_wavs_dir / f"{utt_id}.wav"
    write(writefilename, rate=22050, data = xx)


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    with open(args.utt_list) as f:
        utt_ids = [utt_id.strip() for utt_id in f]
    wav_files = [Path(args.wav_root) / f"{utt_id}.wav" for utt_id in utt_ids]
    lab_files = [Path(args.lab_root) / f"{utt_id}.lab" for utt_id in utt_ids]

    in_dir = Path(args.out_dir) / "in_non_ar_asr"
    out_dir = Path(args.out_dir) / "out_non_ar_asr"
    dur_dir = Path(args.out_dir) / "dur_non_ar_asr"
    hi_mels_dir = Path(args.hi_root) / "JSUT" / "mels"
    hi_wavs_dir = Path(args.hi_root) / "JSUT" / "wavs"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    dur_dir.mkdir(parents=True, exist_ok=True)
    hi_mels_dir.mkdir(parents=True, exist_ok=True)
    hi_wavs_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(args.n_jobs) as executor:
        futures = [
            executor.submit(
                preprocess,
                wav_file,
                lab_file,
                args.sample_rate,
                args.n_fft,
                args.n_mels,
                args.win_length,
                args.hop_length,
                args.f_min,
                args.f_max,
                in_dir,
                out_dir,
                dur_dir,
                hi_mels_dir,
                hi_wavs_dir
            )
            for wav_file, lab_file in zip(wav_files, lab_files)
        ]
        for future in tqdm(futures):
            future.result()
            #print( future )


   
