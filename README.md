# MULTI-SPEAKER GLOW
Tilde's multi speaker extension to [Glow-TTS](https://github.com/jaywalnut310/glow-tts). Currently only supports GPU training, but can perform CPU/GPU waveglow synthesis.

## Installation for UNIX
1. `pip install -r requirements.txt`
2. `pip install cython`
3. `cd glow-tts/; ./build_glow_cython.sh`
4. Install Apex

## Training calls
Similar functionality to GLOW model. Config file has been switched to YAML and modified to include directory of the embedding vectors.
Embedding vectors <em>MUST</em> have the same name as their audio counterparts (except for the extension). Refer to `configs/config_glow.yaml`.

1. `./train_ddi.sh`

## Synthesis calls
Currently requires precomputed speaker embeddings for the desired speaker, which must be provided in the input file by means of audio path.
Supports only WAVEGLOW neural vocoder.
1. `python synth.py -f path/to/sample.txt -c path/to/checkpoint.pth -hp path/to/config_glow.yaml -o outdir -w path/to/waveglow_ckpt`
2. (OPTIONAL) add `--cuda` to use GPU
3. (OPTIONAL) add `--spaces` to append start/end spaces to text (may improve synthesis quality)


sample.txt layout
`path/to/audio.wav|desired text to be synthesized`

## Differences from single speaker GLOW
Contrary to the suggestions in the original [GLOW paper](https://arxiv.org/pdf/2005.11129.pdf), speaker embeddings are concatenated to the output of the text encoder.
This way the prior distribution depends on the speaker identity and speaker embeddings have influence on the most probable alignment with the latent representation.

## Obtaining speaker embeddings
Speaker embeddings can be extracted using a speaker verification network, which learns to discriminate between speakers. The quality of the embeddings can be measured using
dimensionality reduction techniques (such as PCA) and observing the resulting clusters. Individual speakers should appear clustered and there should be clear separation between the genders. 
It is also possible to use a pretrained model, which, if sufficiently large, can come from a different language. To extract from Kaldi xvec network trained on VOXceleb dataset, refer to [KALDI recipe](https://github.com/kaldi-asr/kaldi/tree/master/egs/voxceleb/v2).

# ACKNOWLEDGMENTS
The research has been supported by the European Regional Development Fund within the research project ”Multilingual Artificial Intelligence Based Human Computer Interaction” No. 1.1.1.1/18/A/148
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>

# BELOW FOLLOWS THE ORIGINAL GLOW DESCRIPTION

# Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search

### Jaehyeon Kim, Sungwon Kim, Jungil Kong, and Sungroh Yoon

In our recent [paper](https://arxiv.org/abs/2005.11129), we propose Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search.

Recently, text-to-speech (TTS) models such as FastSpeech and ParaNet have been proposed to generate mel-spectrograms from text in parallel. Despite the advantages, the parallel TTS models cannot be trained without guidance from autoregressive TTS models as their external aligners. In this work, we propose Glow-TTS, a flow-based generative model for parallel TTS that does not require any external aligner. We introduce Monotonic Alignment Search (MAS), an internal alignment search algorithm for training Glow-TTS. By leveraging the properties of flows, MAS searches for the most probable monotonic alignment between text and the latent representation of speech. Glow-TTS obtains an order-of-magnitude speed-up over the autoregressive TTS model, Tacotron 2, at synthesis with comparable speech quality, requiring only 1.5 seconds to synthesize one minute of speech in end-to-end. We further show that our model can be easily extended to a multi-speaker setting.

Visit our [demo](https://jaywalnut310.github.io/glow-tts-demo/index.html) for audio samples.

We also provide the [pretrained model](https://drive.google.com/open?id=1JiCMBVTG4BMREK8cT3MYck1MgYvwASL0).

<table style="width:100%">
  <tr>
    <th>Glow-TTS at training</th>
    <th>Glow-TTS at inference</th>
  </tr>
  <tr>
    <td><img src="resources/fig_1a.png" alt="Glow-TTS at training" height="400"></td>
    <td><img src="resources/fig_1b.png" alt="Glow-TTS at inference" height="400"></td>
  </tr>
</table>

## 1. Environments we use

* Python3.6.9
* pytorch1.2.0
* cython0.29.12
* librosa0.7.1
* numpy1.16.4
* scipy1.3.0

For Mixed-precision training, we use [apex](https://github.com/NVIDIA/apex); commit: 37cdaf4


## 2. Pre-requisites

a) Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/), then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY`


b) Initialize WaveGlow submodule: `git submodule init; git submodule update`

Don't forget to download pretrained WaveGlow model and place it into the waveglow folder.

c) Build Monotonic Alignment Search Code (Cython): `cd monotonic_align; python setup.py build_ext --inplace`


## 3. Training Example


```sh
sh train_ddi.sh configs/base.json base
```

## 4. Inference Example

See [inference.ipynb](./inference.ipynb)


## Acknowledgements
Our implementation is highly affected by the following repos:
* [WaveGlow](https://github.com/NVIDIA/waveglow)
* [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)
* [Mellotron](https://github.com/NVIDIA/mellotron)
