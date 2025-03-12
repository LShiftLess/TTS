All files in this folder should be used in terminal with opening the parent folder (It defaults to 'TTS').

To train a tts model, simply run the following prompt.

```bash
# Basic
python TTS/bin/train_tts.py --gpu 0 --config_path rotaterm/config.json
# Train a pretrained model
python TTS/bin/train_tts.py --gpu 0 --config_path rotaterm/config.json --restore_path ${pretrained_model_path}
# Continue training from a ckpt
python TTS/bin/train_tts.py --gpu 0 --continue_path ${ckpt_path}
```