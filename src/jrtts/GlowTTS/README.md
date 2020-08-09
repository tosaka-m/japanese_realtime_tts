# GlowTTS

This implementation uses code from the following repo:
- [glow-tts](https://github.com/jaywalnut310/glow-tts)

## Requirements

- pytorch==1.5
- torchaudio==0.5.0
- python>=3.6
- wandb


## Training

```
python train.py --config_path Configs/config.yml [-t]
```

## Inference
see Notebooks/inference.ipynb


## Config

|Name|Description|type|
|log_dir|log out dirname|str|
|save_freq|save model per epoch|int|
|device|set cpu or cuda. only support single gpu training|str|
|batch_size|train batch size|int|
|pretrained_model|pretrained model relative path|str|
|dataset_params|kwargs of Utils.build_dataloader.FilePathDataset|dict|
|model_params|kwargs of Networks.models.FlowGenerator|dict|
|optimizer_params|kwargs of Utils.build_optimizer.build_optimizer|dict|
