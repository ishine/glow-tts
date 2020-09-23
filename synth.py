import os

from text.symbols import symbols
import models
import utils

import sys
import argparse

import numpy as np
import torch

from scipy.io.wavfile import write
from text import text_to_sequence
from waveglow.denoiser import Denoiser


def load_waveglow_model(model_path: str, device: torch.device):
    # this is required for pickle to see glow module
    sys.path.append("tts_dev/waveglow/")

    waveglow = torch.load(model_path, map_location=device)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)

    waveglow.eval()
    if device.type != 'cpu':
        waveglow.cuda().half()
    for k in waveglow.convinv:
        k.float()

    denoiser = Denoiser(waveglow)

    return waveglow, denoiser


def synthesize_glow(model, device: torch.device, hparams, input_file: str, output_dir: str, waveglow, denoiser, spaces):
    speakers = []
    audio_names = []
    sampling_rate = hparams.data.sampling_rate
    file = open(input_file, "r", encoding="utf-8")
    i = 0
    for line in file:

        audio_path = os.path.join(output_dir, "synthesized_audio_{}.wav".format(i))

        line = line.split("|")
        # embedding part
        filename = line[0]
        speakers.append(filename)
        audio_names.append("audio_" + str(i))
        filename = filename[filename.rfind("/"):]
        filename = filename.replace(".wav", ".npy")
        # store speaker name for PCA
        # construct correct path
        filename = hparams.data.speaker_id_dir + filename
        speaker_embedding = np.load(filename)
        speaker_embedding = torch.from_numpy(speaker_embedding)
        # missing stack action from data_utils, hence an extra unsqueeze
        speaker_embedding = speaker_embedding.unsqueeze(0)
        speaker_embedding = speaker_embedding.float().to(device)

        # text part
        text = line[1]
        if spaces:
            text = " " + text + " "
        sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
        print("Input: ", "".join([symbols[c] for c in sequence[0]]))

        x_tst = torch.from_numpy(sequence).long().to(device)
        x_tst_lengths = torch.tensor([x_tst.shape[1]]).to(device)
        with torch.no_grad():
            noise_scale = .667
            length_scale = 1.0
            (y_gen_tst, *r), attn_gen, *_ = model(x_tst, x_tst_lengths, speaker_embedding, gen=True,
                                                  noise_scale=noise_scale,
                                                  length_scale=length_scale)
        # waveglow "vocoder"
        import apex as amp
        waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")  # Try if you want to boost up synthesis speed.
        try:
            audio = waveglow.infer(y_gen_tst.half(), sigma=.666)
        except:
            audio = waveglow.infer(y_gen_tst, sigma=.666)
        audio = denoiser(audio, strength=0.01)[:, 0]

        write(audio_path, sampling_rate,
              audio[0].clamp(-1, 1).data.cpu().float().numpy())

        i += 1
        print(audio_path)
    return speakers, audio_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--input_file', type=str, help='Input file with text inside', required=True)
    parser.add_argument("-c", "--checkpoint_glow", type=str, default=None, required=True,
                        help="Path to glow checkpoint.")
    parser.add_argument("-hp", "--hyperparams", type=str, default=None, required=True,
                        help="Path to config file in JSON format")
    parser.add_argument("-o", "--output_dir", type=str, default=None, required=True,
                        help="Output directory path, where plots and wavs will be put.")
    parser.add_argument("--cuda", action='store_true', help="Add to run on gpu")
    parser.add_argument("--spaces", action='store_true', help="Add to add start/end spaces for glow synthesis")
    parser.add_argument("-w", "--waveglow_path", type=str, default=None, required=True,
                        help="Path to waveglow checkpoint")
    args = parser.parse_args()

    # set device
    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # loading models
    print("Loading models...")

    hps = utils.get_hparams_from_dir(args.hyperparams)
    model = models.FlowGenerator(
        speaker_dim=hps.model.speaker_embedding,
        n_vocab=len(symbols),
        out_channels=hps.data.n_mel_channels,
        **hps.model).to(device)
    utils.load_checkpoint(args.checkpoint_glow, model)
    model.decoder.store_inverse()  # do not calcuate jacobians for fast decoding
    _ = model.eval()
    print("---GLOW--- loaded")
    # handle case of no path
    waveglow, denoiser = load_waveglow_model(args.waveglow_path, device)
    print("Using waveglow neural vocoder")
    # synthesis
    print("Synthesizing...")
    speakers, audio_names = synthesize_glow(model, device, hps, args.input_file, args.output_dir, waveglow, denoiser,
                                            args.spaces)
    print("Speech synthesis complete.")


if __name__ == '__main__':
    main()
