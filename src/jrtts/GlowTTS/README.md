# GlowTTS

This implementation uses code from the following repo:
- [glow-tts](https://github.com/jaywalnut310/glow-tts)

## Requirements
- python>=3.6

- pytorch==1.5
- torchaudio==0.5.0
- pyworld
- soundfile
- librosa
- wandb

## Preprocessing
1. 以下の形式の train, val データを用意する
```
wavfile_path1|text1|speakerid1
wavfile_path2|text2|speakerid2
wavfile_path3|text3|speakerid3
...
```
- Data/train_list.txt, Data/val_list.txt 参照
- Single Speaker なら speakerid は何でも良い
- jsut を利用する場合は Data/wavefiles 以下に保存

2. config ファイルを作成する

```
cp Configs/base_config.yml Configs/config.yml
```

config.yml の各項目を必要に応じて修正する。


3. 以下を実行する

```
python preprocess.py --config Configs/config.yml [--filelists Data/train_list.txt Data/val_list.txt] [--use_pitch] [--out_dir Data/processed]
```
- 別のデータファイルリストを用意した場合は --filelists で指定する
- pitch を使用する場合は --use_pitch を入れる
- 出力先を変更する場合は --out_dir を変更する

## Training
1. wandb に登録し以下を実行

```
wandb init
```

1. pretrained model をダウンロード
- glow-tts official pretrained model
- my jsut pretrained model

2. 以下を実行
```
python train.py --config_path Configs/config.yml [-t]
```
- [-t] はテスト時用



## Inference
see Notebooks/inference.ipynb


## Config


|Name|Description|type|
|:---|:---|:---|
|log_dir|log out dirname|str|
|save_freq|model is saved every save_freq epoch|int|
|device|set cpu or cuda. only support single gpu training|str|
|batch_size|train batch size|int|
|pretrained_model|pretrained model relative path|str|
|dataset_params|kwargs of Utils.build_dataloader.FilePathDataset|dict|
|model_params|kwargs of Networks.models.FlowGenerator|dict|
|optimizer_params|kwargs of Utils.build_optimizer.build_optimizer|dict|
