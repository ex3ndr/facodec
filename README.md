# ✨ FACodec for Pytorch (Torch Hub)

This is easy to use FACodec model for voice encoding, which can be used for voice conversion, voice style transfer, and voice synthesis. The model is exported from [official implementation](https://github.com/open-mmlab/Amphion/tree/main/models/codec/ns3_codec) and uses [original weights](https://huggingface.co/amphion/naturalspeech3_facodec). This library just wraps the model and provides an easy to use interface, without the need to mangle with the original code.

## Installation

This library is meant to be used with Torch Hub and depends only on `torch` and `torchaudio` packages:

```python
facodec = torch.hub.load(repo_or_dir='ex3ndr/facodec', model='facodec', trust_repo = True)
```

## Evaluation

To evaluate model you can use [evaluation notebook](/eval.ipynb) which can run anywhere where `torch` and `torchaudio` are installed.

## How to use

FACodec expects a `1D` tensor of a waveform in 16khz, normalized in `[-1, 1]` range.

```python
import torch
import torchaudio

#
# Loading Audio
#
def load_mono_audio(path):
    # Load audio
    audio, sr = torchaudio.load(path)

    # Resample
    if sr != 16000:
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)
        sr = 16000

    # Convert to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Convert to single dimension
    audio = audio[0]

    return audio

source = load_mono_audio("example.wav")

#
# Encode source audio
# 

prosody_code, cotent_code, residual_code, spk_embs = facodec.encode(source)
print("Speaker Embedding: ", spk_embs.shape) # [256]
print("Prosody Code: ", prosody_code.shape) #[1, N]
print("Content Code: ", cotent_code.shape) #[2, N]
print("Residual Code: ", residual_code.shape) #[3, N]

#
# Decode back
#

reconstructed = facodec.decode(prosody_code, cotent_code, residual_code, spk_embs)

#
# Speaker style changing
#

speaker_sample = load_mono_audio("./eval/eval_2.mp3")
speaker_embedding = facodec.speaker_embedding(speaker_sample)
reconstructed_styled = facodec.speech_convert(prosody_code, cotent_code, speaker_embedding)

```

## License

MIT