import h5py
import json
import numpy as np
from pathlib import Path
import pickle
import random
import librosa

import torch
from torch.utils.data import Dataset

from utils.util import load_dict_from_csv


class AudioTextExpertDataset(Dataset):
    """
    Available audio features:
        - VGGish pretrained feature: vggish
        - Resnet18 VGGSound pretrained feature: vggsound
        - PANNs pretrained feature: panns_cnn10, panns_cnn14
    w2v embedding pretrained by googleNews-vectors-negative300

    :Params: w2v_file: filepath to w2v pickle file 
             audio_feature: list of audio feature dict, List[Dict]
             audio_experts: list of audio_experts, List[str]
             filename: filepath to index file
             split: datasplit train, val or test

    """

    def __init__(self,
                 audio_features,
                 audio_experts,
                 text_feature,
                 filename,
                 max_words,
                 audio_padding_length,
                 split):
        self.modalities = audio_experts
        self.audio_features = audio_features
        self.num_audio_features = len(audio_features)
        self.audio_feature_cache = {}
        self.text_feature = text_feature
        self.text_feature_cache = {}
        self.fname = json.load(open(filename, "r"))["audios"]
        self.split = split
        self.max_words = max_words
        self.audio_padding_length = audio_padding_length

    def __len__(self):
        return len(self.fname)

    def get_audio_feature(self, audio_id):
        audio_features = {}
        audio_masks = {}
        for i, mod in enumerate(self.modalities):
            feature_file = self.audio_features[i][audio_id]
            if not feature_file in self.audio_feature_cache:
                self.audio_feature_cache[feature_file] = h5py.File(feature_file, "r")
            audio_feature = self.audio_feature_cache[feature_file][audio_id][()]
            audio_features[mod] = np.zeros((self.audio_padding_length[mod], audio_feature.shape[1]))
            if audio_feature.shape[0] <= self.audio_padding_length[mod]:
                audio_features[mod][:audio_feature.shape[0],:] = audio_feature
            else:
                audio_features[mod] = audio_feature[:self.audio_padding_length[mod],:]
            audio_masks[mod] = [1] * audio_feature.shape[0]
            while len(audio_masks[mod]) < self.audio_padding_length[mod]:
                audio_masks[mod].append(0)
            if len(audio_masks[mod]) > self.audio_padding_length[mod]:
                audio_masks[mod] = audio_masks[mod][:self.audio_padding_length[mod]]
            assert len(audio_masks[mod]) == self.audio_padding_length[mod]
            assert audio_features[mod].shape[0] == self.audio_padding_length[mod]
            
            audio_features[mod] = torch.from_numpy(audio_features[mod]).float()
            audio_masks[mod] = torch.as_tensor(audio_masks[mod])
        return audio_features, audio_masks

    def get_text_feature(self, audio_id):
        feature_file = self.text_feature[audio_id]
        if not feature_file in self.text_feature_cache:
            self.text_feature_cache[feature_file] = pickle.load(
                open(feature_file, 'rb'))
        captions = self.text_feature_cache[feature_file][audio_id]
        text_features = []
        max_token_masks = []
        for i in range(len(captions)):
            caption = captions[i]
            max_token_masks.append([1] * caption.shape[0])
            text_features.append(np.zeros((self.max_words, caption.shape[-1])))
            if caption.shape[0] <= self.max_words:
                text_features[i][:caption.shape[0]] = caption
            else:
                text_features[i] = caption[:self.max_words,:]
            
            while len(max_token_masks[i]) < self.max_words:
                max_token_masks[i].append(0)
            if len(max_token_masks[i]) > self.max_words:
                max_token_masks[i] = max_token_masks[i][:self.max_words]
            assert len(max_token_masks[i]) == self.max_words
            assert text_features[i].shape[0] == self.max_words
            
            text_features[i] = torch.from_numpy(text_features[i]).float()
            max_token_masks[i] = torch.as_tensor(max_token_masks[i])

        text_features = torch.stack(text_features)
        max_token_masks = torch.stack(max_token_masks)  
        return text_features, max_token_masks
    
    def __getitem__(self, idx):
        audio_id = self.fname[idx]['audio_id']
        audio_features, audio_masks = self.get_audio_feature(audio_id)
        text_features, max_token_masks = self.get_text_feature(audio_id)
        ind = {mod: torch.ones(1) for mod in self.modalities}
        return {
            'experts': audio_features,
            'text': text_features,
            'expert_masks': audio_masks,
            'text_token_masks': max_token_masks,
            'ind': ind,
            'aid': audio_id
        }


class AudioTextDataset(Dataset):

    def __init__(self,
                 audio_file,
                 text_file,
                 random_subset=None,
                 source_sample_rate: int = 32000,
                 target_sample_rate: int = 32000,
                 audio_duration=None,
                 random_subsample=False):
        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self.audio_cache = {}
        self.aid_to_waveform = load_dict_from_csv(audio_file)
        data = json.load(open(text_file, "r"))["audios"]
        self.text_data = {item["audio_id"]: item["captions"] for item in data}
        if random_subset:
            aids = random.sample(self.text_data.keys(), random_subset)
            self.text_data = {k: v for k, v in self.text_data.items() 
                if k in aids}
        self.aids = list(self.text_data.keys())
        self.audio_duration = audio_duration
        if audio_duration:
            self.audio_duration = audio_duration * target_sample_rate
            self.random_subsample = random_subsample

    def __len__(self):
        return len(self.aids)

    def encode_text(self, audio_id, text):
        return text

    def get_audio(self, audio_id):
        try:
            hdf5_path = self.aid_to_waveform[audio_id]
        except KeyError:
            audio_id = "Y" + audio_id + ".wav"
            hdf5_path = self.aid_to_waveform[audio_id]
        if not hdf5_path in self.audio_cache:
            self.audio_cache[hdf5_path] = h5py.File(hdf5_path, "r")
        waveform = np.array(self.audio_cache[hdf5_path][audio_id][()],
            dtype=np.float32)
        waveform = librosa.core.resample(waveform, self.source_sample_rate,
            self.target_sample_rate)
        return waveform

    def get_text(self, audio_id):
        data = self.text_data[audio_id]
        text = [item["caption"] for item in data]
        return self.encode_text(audio_id, text)
    
    def __getitem__(self, idx):
        audio_id = self.aids[idx]
        waveform = self.get_audio(audio_id)
        if self.audio_duration and waveform.shape[0] > self.audio_duration:
            if self.random_subsample:
                start = 0
            else:
                start = random.randint(0, waveform.shape[0] - self.audio_duration)
            waveform = waveform[start: start + self.audio_duration]
        text = self.get_text(audio_id)
        text = self.encode_text(audio_id, text)
        return {
            "aid": audio_id,
            "waveform": waveform,
            "text": text,
        }



class AudioSingleTextDataset(AudioTextDataset):

    def __init__(self, audio_file, text_file, source_sample_rate: int = 32000,
        target_sample_rate: int = 32000, random_subset=None, audio_duration=None):
        super().__init__(audio_file, text_file, random_subset,
            source_sample_rate, target_sample_rate)
        text_data = []
        for aid, cap_items in self.text_data.items():
            for cap_item in cap_items:
                text_data.append(
                    (aid, cap_item["caption"])
                )
        self.text_data = text_data
        self.audio_duration = audio_duration
        if audio_duration:
            self.audio_duration = audio_duration * target_sample_rate

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        aid, text = self.text_data[idx]
        waveform = self.get_audio(aid)
        if self.audio_duration and waveform.shape[0] > self.audio_duration:
            start = random.randint(0, waveform.shape[0] - self.audio_duration)
            waveform = waveform[start: start + self.audio_duration]
        return {
            "aid": aid,
            "waveform": waveform,
            "text": text
        }


def collate_fn(list_data_dict):
    waveform, wave_length, text, aid = [], [], [], []
    for data_dict in list_data_dict:
        waveform.append(data_dict["waveform"])
        wave_length.append(data_dict["waveform"].shape[0])
        text.append(data_dict["text"])
        aid.append(data_dict["aid"])
    waveform = torch.as_tensor(np.array(waveform)).float()
    wave_length = torch.as_tensor(wave_length).int()
    text = torch.as_tensor(np.array(text)).float()
    aid = np.array(aid)
    return {
        "waveform": waveform,
        "wave_length": wave_length,
        "text": text,
        "aid": aid
    }


def collate_fn_transformers(tokenizer_type, max_text_length):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
    def wrapper(list_data_dict):
        waveform, wave_length, text, aid = [], [], [], []
        for data_dict in list_data_dict:
            waveform.append(data_dict["waveform"])
            wave_length.append(data_dict["waveform"].shape[0])
            text.append(data_dict["text"])
            aid.append(data_dict["aid"])
        aid = np.array(aid)
        batch_size = aid.shape[0]
        tmp = np.empty((batch_size, max(wave_length)))
        for idx, wav in enumerate(waveform):
            tmp[idx, :len(wav)] = wav
        waveform = torch.as_tensor(tmp).float()
        wave_length = torch.as_tensor(wave_length).int()
        text = [data_dict["text"] for data_dict in list_data_dict]
        num_captions = len(text[0])
        text = sum(text, [])
        tokens = dict(tokenizer(text, padding="max_length",
            max_length=max_text_length, truncation=True, return_tensors="pt"))
        for k in tokens:
            tokens[k] = tokens[k].view(batch_size, num_captions, *tokens[k].size()[1:])
        output = {
            "waveform": waveform,
            "wave_length": wave_length,
            "aid": aid,
            "num_captions": num_captions
        }
        output.update(tokens)
        return output
    return wrapper


def collate_fn_single_text(tokenizer_type, max_text_length):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
    def wrapper(list_data_dict):
        waveform, wave_length, text, aid = [], [], [], []
        for data_dict in list_data_dict:
            waveform.append(data_dict["waveform"])
            wave_length.append(data_dict["waveform"].shape[0])
            text.append(data_dict["text"])
            aid.append(data_dict["aid"])
        aid = np.array(aid)
        batch_size = aid.shape[0]
        tmp = np.empty((batch_size, max(wave_length)))
        for idx, wav in enumerate(waveform):
            tmp[idx, :len(wav)] = wav
        waveform = torch.as_tensor(tmp).float()
        wave_length = torch.as_tensor(wave_length).int()
        text = [data_dict["text"] for data_dict in list_data_dict]
        tokens = dict(tokenizer(text, padding="max_length",
            max_length=max_text_length, truncation=True, return_tensors="pt"))
        output = {
            "waveform": waveform,
            "wave_length": wave_length,
            "aid": aid,
        }
        output.update(tokens)
        return output
    return wrapper


