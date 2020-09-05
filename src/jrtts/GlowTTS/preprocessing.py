'''
compute mel and f0 from wav

- Only Support wav file
'''
import os
import os.path as osp
import yaml
import click
import numpy as np
import torchaudio
import torch
from tqdm import tqdm
import soundfile as sf

@click.command()
@click.option('--config_path', '-p', default='Configs/config.yml')
@click.option('--filelists', '-f', default=["Data/train_list.txt"], multiple=True)
@click.option('--use_pitch', is_flag=True)
@click.option('--out_dir', default='Data/processed')
def main(config_path, filelists, use_pitch, out_dir):
    if not osp.exists(out_dir): os.mkdir(out_dir)

    with open('Configs/base_config.yml') as f:
        config = yaml.safe_load(f)
    with open(config_path) as f:
        update_config = yaml.safe_load(f)
        config.update(update_config)

    dataset_params = config['dataset_params']
    sr = dataset_params['sampling_rate']
    to_melspec = torchaudio.transforms.MelSpectrogram(
        n_fft=dataset_params['n_fft'],
        win_length=dataset_params['win_length'],
        hop_length=dataset_params['hop_length'],
        n_mels=dataset_params['n_mels'],
    )

    for filelist_path in filelists:
        out_listpath = filelist_path + '.processed'
        with open(filelist_path) as f:
            lines = f.readlines()
        data_paths = [line.split('|')[0] for line in lines]
        common_path = osp.commonpath(data_paths)
        outpaths = [osp.join(
            out_dir, osp.relpath(path.replace('.wav', '_wav.pth'), common_path)) for path in data_paths]

        for idx, (path, outpath) in enumerate(tqdm(zip(data_paths, outpaths), desc='[preprocess]'), 1):
            preprocess(path, outpath, to_melspec, sr, use_pitch)

        with open(out_listpath, 'w') as f:
            for line, outpath in zip(lines, outpaths):
                out_line = '|'.join([outpath] + line.split('|')[1:])
                if osp.exists(outpath):
                    f.write(out_line)

def preprocess(path, outpath, to_melspec, sr, use_pitch=True):
    # hyperparams
    hop_length = to_melspec.hop_length
    n_mels = to_melspec.n_mels
    thresh = 5.5 # remove audio threshold
    fix_duration = 2400 # =0.1sec, margin

    outdir = osp.dirname(outpath)
    os.makedirs(outdir, exist_ok=True)

    # load and normalize wav
    wav, _sr = sf.read(path)
    if sr != _sr:
        import librosa
        print('Resample from %d to %d' % (_sr, sr))
        wav = librosa.resample(wav, _sr, sr)

    wav = torch.from_numpy(wav).float()
    wav = (wav / wav.abs().max()) * 0.99
    if len(wav) < sr:
        print('Too short wav file: %d: %s' % (wav.shape[0], path))
        return 0

    # cut nosignal
    mel = to_melspec(wav)
    power = torch.log(mel.norm(dim=0))
    min_index = max(0, (np.where(power.numpy() > thresh)[0][0]) * hop_length - fix_duration)
    max_index = (np.where(power.numpy() > thresh)[0][-1]) * hop_length + fix_duration

    rewav = wav[min_index:max_index]
    remel = to_melspec(rewav)
    size_ratio = rewav.shape[0] / wav.shape[0]
    if use_pitch:
        f0 = get_f0(rewav.numpy(), sr, frame_period=1000*hop_length/sr)
        f0 = torch.from_numpy(f0).float()
    else:
        f0 = torch.FloatTensor([0]) #dummy f0

    assert(remel.shape[1] == f0.shape[0])
    torch.save([rewav, remel, f0], outpath)

def get_f0(wave, sr=24000, frame_period=12.5, method='harvest'):
    import pyworld
    _wave = wave.astype(np.float64)
    if method == 'harvest':
        _f0, _time = pyworld.harvest(_wave, sr, frame_period=frame_period)
    elif method == 'dio':
        _f0, _time = pyworld.harvest(_wave, sr, frame_period=frame_period)
    else:
        raise RuntimeError('Got unexpected method %s, expect harvest or dio' % method)
    f0 = pyworld.stonemask(_wave, _f0, _time, sr)
    return f0


if __name__=='__main__':
    main()
