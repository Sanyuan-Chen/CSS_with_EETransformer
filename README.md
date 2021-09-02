# Multi-channel Continuous Speech Separation with Early Exit Transformer

## Introduction

We elaborate an early exit mechanism for Transformer based multi-channel speech separation, which aims to address the “overthinking” problem and accelerate inference stage simultaneously.

For a detailed description and experimental results, please refer to our paper: [Don't shoot butterfly with rifles: Multi-channel Continuous Speech Separation with Early Exit Transformer](https://arxiv.org/abs/2010.12180) (Accepted by ICASSP 2021).

## Environment
python 3.6.9, torch 1.7.1

## Get Started
1. Download the overlapped speech of [LibriCSS dataset](https://github.com/chenzhuo1011/libri_css).

    ```bash
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1PdloA-V8HGxkRu9MnT35_civpc3YXJsT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1PdloA-V8HGxkRu9MnT35_civpc3YXJsT" -O overlapped_speech.zip && rm -rf /tmp/cookies.txt && unzip overlapped_speech.zip && rm overlapped_speech.zip
   ```

2. Download the Conformer separation models.

    ```bash
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bK_0jj4yQjCJUOX-Bd8x_1PJNQL8UvfZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bK_0jj4yQjCJUOX-Bd8x_1PJNQL8UvfZ" -O checkpoints.zip && rm -rf /tmp/cookies.txt && unzip checkpoints.zip && rm checkpoints.zip
    ```

3. Run the separation.
    
    ```bash
    export MODEL_NAME=EETransformer
    export EE_THRESHOLD=0
    python3 separate.py \
        --checkpoint checkpoints/$MODEL_NAME \
        --mix-scp utils/overlapped_speech_7ch.scp \
        --dump-dir separated_speech/7ch/utterances_with_${MODEL_NAME}_eet${EE_THRESHOLD} \
        --device-id 0 \
        --num_spks 2 \
        --mvdr True \
        --early_exit_threshold $EE_THRESHOLD
    ```
    
    The separated speech can be found in the directory 'separated_speech/7ch/utterances_with_${MODEL_NAME}_eet${EE_THRESHOLD}'

## Citation
If you find our work useful, please cite [our paper](https://arxiv.org/abs/2010.12180):
```bibtex
@inproceedings{CSS_with_EETransformer,
  title={Don’t shoot butterfly with rifles: Multi-channel continuous speech separation with early exit transformer},
  author={Chen, Sanyuan and Wu, Yu and Chen, Zhuo and Yoshioka, Takuya and Liu, Shujie and Li, Jinyu and Yu, Xiangzhan},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={6139--6143},
  year={2021},
  organization={IEEE}
}
```
