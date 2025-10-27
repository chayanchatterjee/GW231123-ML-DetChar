# GW231123-ML-DetChar
This repository contains training and analysis scripts for the paper,["Machine Learning Confirms GW231123 is a "Lite" Intermediate-Mass Black Hole Merger"](https://arxiv.org/abs/2509.09161). In this work, we have performed machine learning-based event validation of GW231123 - the most massive binary black hole merger detected by LIGO-Virgo-KAGRA Collaboration. We have used three ML models for our analyses: GW-Whisper, an innovative application of [OpenAI](https://openai.com/)’s [Whisper model](https://arxiv.org/abs/2212.04356), originally designed for speech recognition, to gravitational wave data analysis, ArchGEM, a Gaussian mixture model for scattering glitch characterization, and [AWaRe](https://github.com/chayanchatterjee/AWaRe) or Attention-boosted Waveform Reconstruction network, an Encoder-Decoder network that produces reconstructions of gravitational wave signals with associated uncertainties. 


This repository is a work in progress. Details about dependencies, installation and execution of the scripts will be added soon.

## Components

| Repository | Description |
|------------|---------|
| [GW-Whisper](https://github.com/chayanchatterjee/GW231123-ML-DetChar/tree/main/GW-Whisper) | Code for GW-Whisper based classification and search |
| ArchGEM | Code repo for ArchGEM scattering glitch analysis |
| [AWaRe](https://github.com/chayanchatterjee/GW231123-ML-DetChar/tree/main/AWaRe)|  | Code for waveform reconstruction tests using AWaRe|


