# Audio-text Retrieval for DCASE2022 Challenge

## Set up environment
```bash
$ git clone https://github.com/wsntxxn/DCASE2022T6_CLAP
$ cd DCASE2022T6_CLAP
$ pip install -r requirements.txt
```

## Prepare the dataset

We use [Clotho](https://arxiv.org/abs/1910.09387), [AudioCaps](https://www.aclweb.org/anthology/N19-1011/) and [MACS](https://trepo.tuni.fi/bitstream/handle/10024/137744/Diversity_and_bias.pdf) for training. Due to the copyright problem of Youtube videos, we do not include the step-by-step data preparation scripts in this repo. However, once the audio and text files are downloaded, you can use the tools to convert the data into the desired format.

The recipe in [the audio captioning repo](https://github.com/wsntxxn/AudioCaption/tree/master/data) can help.

First organize audio files to generate `wav.csv`. It is a tab separated table like this:

|audio_id|file_name|
|----:|-----:|
|Y_f_i28HxMDA.wav|/path/to/audioset/Y_f_i28HxMDA.wav|
|Ya67ihfUaKUc.wav|/path/to/audioset/Ya67ihfUaKUc.wav|
|...  |...   |

`data/utils/prepare_wav_csv.py` can be used to generate `wav.csv` for a single dataset:
```bash
$ python data/utils/prepare_wav_csv.py $AUDIO_DIR $WAV_CSV
```
where `$AUDIO_DIR` contains all .wav audio files of a dataset, and `$WAV_CSV` is the path of the output `wav.csv`.
Then `data/utils/pack_waveform.py` packs the .wav files into a single hdf5 file:
```bash
$ python data/utils/pack_waveform.py $WAV_CSV -o $HDF5_FILE --sample_rate $SAMPLE_RATE
```

Text annotations should be converted into the json format. 

Finally hdf5 files and annotation files are concatenated into a single file by `data/utils/concat_csv.py` and `data/utils/concat_json.py`.
During training we randomly select 1000 audio-text pairs as the validation set, which is done by `data/utils/train_val_split.py`.

## Prepare the pre-trained models
We use several pre-trained audio encoders. For example, PANNs CNN14:
```bash
$ mkdir pretrained_models
$ wget -O pretrained_models/Cnn14_mAP=0.431.pth https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth
```
Similarly, other models should also be downloaded in advance.


## Prepare the configuration
Example configuration files are in `configs`. Modify the corresponding settings (e.g. data files, training iterations, etc.).

## Training using slurm
The training scripts use slurm wordload manager. It is also based on distributed data parallel. You can modify the script to use your own system for training.
```bash
$ sbatch scripts/run_ddp.sh
```
