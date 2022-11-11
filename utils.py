import os.path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy.io import wavfile
import spacy
import logging
from collections import Counter
import csv
import pickle
import random
import itertools
from sklearn.metrics import f1_score
import yaml
import decord
from decord import VideoReader
from decord import cpu, gpu
from tqdm import tqdm
import json
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# logger.info("----Loading Spacy----")
# spacy_en = spacy.load('en_core_web_sm')
spacy_en = None


# Calculate F1: use scikit learn and use weighted and use
# data like this https://stackoverflow.com/questions/46732881/how-to-calculate-f1-score-for-multilabel-classification
# get statistics


def get_f1(y_pred, y_label):
    f1 = f1_score(np.array(list(itertools.chain.from_iterable(y_pred))),
                    list(itertools.chain.from_iterable(y_label)), average="weighted")
    return f1


def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def numericalize(inputs, vocab=None, tokenize=False):
    # This should be 2 seperate functions
    # Create vocabs for train file
    if vocab == None:
        # check unique tokens
        counter = Counter()
        for i in inputs:
            if tokenize:
                counter.update(tokenizer(i))
            else:
                counter.update([i])

        # Create Vocab
        if tokenize:  # That is we are dealing with sentences.
            vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        else:
            vocab = {}

        vocab.update({j: i + len(vocab) for i, j in enumerate(counter)})

    # Convert tokens to numbers:
    numericalized_inputs = []
    for i in inputs:
        if tokenize:
            # Adding sos and eos tokens before and after tokenized string
            numericalized_inputs.append([vocab["<sos>"]]+[vocab[j] if j in vocab else vocab["<unk>"] for j in
                                         tokenizer(i)]+[vocab["<eos>"]])  # TODO: doing tokenization twice here
        else:
            numericalized_inputs.append(vocab[i])

    return numericalized_inputs, vocab


def collate_fn(batch, device, audio_pad_value, audio_split_samples):
    """
    We use this function to pad the inputs so that they are of uniform length
    and convert them to tensor

    Note: This function would fail while using Iterable type dataset because
    while calculating max lengths the items in iterator will vanish.
    """

    max_audio_len = 0

    batch_size = len(batch)

    for _, audio_clip, _ in batch:
        if len(audio_clip) > max_audio_len:
            max_audio_len = len(audio_clip)

    # We have to pad the audio such that the audio length is divisible by audio_split_samples
    max_audio_len = (int(max_audio_len/audio_split_samples)+1)*audio_split_samples

    video = torch.stack([batch[0][0] for i in batch])
    audio = torch.FloatTensor(batch_size, max_audio_len).fill_(audio_pad_value).to(device)
    label = torch.LongTensor(batch_size).fill_(0).to(device)


    for i, (_, audio_clip, l) in enumerate(batch):
        audio[i][:len(audio_clip)] = audio_clip
        label[i] = int(l)           # Convert from string ('0') to int (0)

    return video, audio, label


class VT_Dataset:
    def __init__(self, audio, text, action, object_, position, wavs_location):
        self.audio = audio
        self.text = text
        self.action = action
        self.object = object_
        self.position = position
        self.wavs_location = wavs_location

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        _, wav = wavfile.read(os.path.join(self.wavs_location,self.audio[item]))
        return wav, self.text[item], self.action[item], self.object[item], self.position[item]

class VA_Dataset(Dataset):
    def __init__(self, root, v_dim=(128, 128, 32), priority='speed', load_audio='spec', spec_size=(128, 128)):
        '''
        Creates dataset based on Video/Audio pairs. Obtains audio by extracting it from video

        Args:
            root - root directory of data
            v_dim - Desired video dimensions in the form (W, H, F)
            priority - Data loading priority. 
                'speed' loads all data up from for faster training
                'space' loads data in __getitem__ in case of memory restrictions
            load_audio - Type of audio data to load.
                'wav' loads from .wav files
                'spec' loads from spectrogram pngs
        '''
        assert priority in ['speed', 'space'], f'Parameter priority should be "speed" or "space", found: {priority}'

        self.v_path = os.path.join(root, 'video')
        self.a_path = os.path.join(root, 'spec')
        self.videos = []
        self.audios = []
        self.v_dim = v_dim
        self.n_frames = v_dim[2]
        self.priority = priority
        self.load_audio = load_audio
        self.preprocess_spec = T.Compose([
            T.Resize(spec_size),
            T.ToTensor()
        ])
        labels = os.path.join(root, 'labels.json')
        with open(labels, 'r') as f:
            self.data = json.load(f)

        # Convert data dict to hold tuples containing (video, audio, label)
        # ( data[item_name] = (video, audio, label) )
        if priority == 'speed':
            for i in tqdm(range(len(self.data)), desc=f'Loading videos'):
                item_name = list(self.data)[i]
                print(item_name)
                video, audio = self.get_video_audio(item_name)
                self.data[item_name] = (video, audio, self.data[item_name])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        print(f"Getting item {idx}...")
        item_name = list(self.data)[idx]
        label = self.data[item_name]

        if self.priority == 'speed':
            return self.data[item_name]
        else:
            video, audio = self.get_video_audio(item_name)
            # print(f'Video: {video.size()}\nAudio: {audio.size()}')
        return video, audio, label      

    def extract_frames(self, video, start=-1, end=-1):
        decord.bridge.set_bridge('torch')
        vr = VideoReader(video, ctx=cpu(0), width=self.v_dim[0], height=self.v_dim[1])
        total_frames = self.n_frames
        gap = int(len(vr) / total_frames)

        if gap != 0:
            if start < 0: start = 0
            if end < 0: end = len(vr)
            i = list(range(start, end, gap))
            frames = vr.get_batch(i) / 255
        else:
            frames = []
            while len(frames) < total_frames:
                frames.append(vr[0] / 255)
            frames = torch.stack(frames)

        return frames[0:self.n_frames]

    def get_video_audio(self, filename_no_ext):
        v = os.path.join(self.v_path, f'{filename_no_ext}.mp4')

        video = self.extract_frames(v)
        if self.load_audio == 'wav':
            a = os.path.join(self.a_path, f'{filename_no_ext}.wav')
            _, audio = wavfile.read(a)
            audio = torch.from_numpy(audio)[:, 0]     # Eliminate one audio channel
        else:
            a = os.path.join(self.a_path, f'{filename_no_ext}.png')
            audio = self.preprocess_spec(Image.open(a))[:3]

        return video.permute(3, 0, 1, 2), audio

def load_csv(path, file_name):
    # Loads a csv and returns columns:
    with open(os.path.join(path, file_name)) as csv_file:
        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader)
        au, t, ac, o, p = [], [], [], [], []
        for row in csv_reader:
            au.append(row[0])
            t.append(row[1])
            ac.append(row[2])
            o.append(row[3])
            p.append(row[4])
    return au, t, ac, o, p


def get_Dataset_and_vocabs(path, train_file_name, valid_file_name, wavs_location):
    train_data = load_csv(path, train_file_name)
    test_data = load_csv(path, valid_file_name)

    vocabs = []  # to store all the vocabs
    # audio location need not to be numercalized
    # audio files will be loaded in Dataset.__getitem__()
    numericalized_train_data = [train_data[0]]  # to store train data after converting string to ints
    numericalized_test_data = [test_data[0]]  # to store test data after converting strings to ints

    for i, (j, k) in enumerate(zip(train_data[1:], test_data[1:])):
        if i == 0:  # We have to only tokenize transcripts which come at 0th position
            a, vocab = numericalize(j, tokenize=True)
            b, _ = numericalize(k, vocab=vocab, tokenize=True)
        else:
            a, vocab = numericalize(j)
            b, _ = numericalize(k, vocab=vocab)
        numericalized_train_data.append(a)
        numericalized_test_data.append(b)
        vocabs.append(vocab)

    train_dataset = Dataset(*numericalized_train_data, wavs_location)
    valid_dataset = Dataset(*numericalized_test_data, wavs_location)

    Vocab = {'text_vocab': vocabs[0], 'action_vocab': vocabs[1], 'object_vocab': vocabs[2], 'position_vocab': vocabs[3]}

    logger.info(f"Transcript vocab size = {len(Vocab['text_vocab'])}")
    logger.info(f"Action vocab size = {len(Vocab['action_vocab'])}")
    logger.info(f"Object vocab size = {len(Vocab['object_vocab'])}")
    logger.info(f"Position vocab size = {len(Vocab['position_vocab'])}")


    # dumping vocab
    with open(os.path.join(path, "vocab"), "wb") as f:
        pickle.dump(Vocab, f)

    return train_dataset, valid_dataset, Vocab


def get_Dataset_and_vocabs_for_eval(path, valid_file_name, wavs_location):
    test_data = load_csv(path, valid_file_name)

    with open(os.path.join(path, "vocab"), "rb") as f:
        Vocab = pickle.load(f)

    numericalized_test_data = [test_data[0], numericalize(test_data[1], vocab=Vocab['text_vocab'], tokenize=True)[0],
                               numericalize(test_data[2], vocab=Vocab['action_vocab'])[0],
                               numericalize(test_data[3], vocab=Vocab['object_vocab'])[0],
                               numericalize(test_data[4], vocab=Vocab['position_vocab'])[0]]

    valid_dataset = Dataset(*numericalized_test_data, wavs_location)

    return valid_dataset, Vocab


def initialize_weights(m):
    # if hasattr(m, 'weight') and m.weight.dim() > 1:
    #     nn.init.xavier_uniform_(m.weight.data)
    for name, param in m.named_parameters():
        if not isinstance(m, nn.Embedding):
            nn.init.normal_(param.data, mean=0, std=0.01)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, train_iterator, optim, clip, device='cuda'):
    model.train()

    epoch_loss = 0

    # Tracking accuracies
    accuracy = []

    # for f1
    y_pred = []
    y_true = []

    for i, (video, audio, label) in enumerate(train_iterator):
        print('HERE')
        # running batch
        video = video.to(device)
        audio = audio.to(device)
        train_result = model(video, audio, label)

        optim.zero_grad()

        loss = train_result["loss"]
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optim.step()

        # Statistics
        epoch_loss += loss.item()
        y_pred.append(train_result['pred'].tolist())
        y_true.append(label.tolist())

        accuracy.append(sum(train_result["pred"] == label) / len(label) * 100)

    y_pred = list(zip(*y_pred))
    y_true = list(zip(*y_true))

    epoch_f1 = get_f1(y_pred, y_true)
    epoch_accuracy = sum(accuracy) / len(accuracy)

    return epoch_loss / len(train_iterator), (epoch_f1, epoch_accuracy)


def evaluate(model, valid_iterator):
    model.eval()

    epoch_loss = 0

    # Tracking accuracies
    accuracy = []

    # for f1
    y_pred = []
    y_true = []

    with torch.no_grad():
        for i, batch in enumerate(valid_iterator):
            # running batch
            valid_result = model(*batch)

            loss = valid_result["loss"]

            # Statistics
            epoch_loss += loss.item()

            y_pred.append(valid_result["pred"].tolist())
            y_true.append(batch[2].tolist())

            accuracy.append(sum(valid_result["pred"] == batch[2]) / len(batch[2]) * 100)

    y_pred = list(zip(*y_pred))
    y_true = list(zip(*y_true))

    epoch_f1 = get_f1(y_pred, y_true)
    epoch_action_accuracy = sum(accuracy) / len(accuracy)

    return epoch_loss / len(valid_iterator), (epoch_f1, accuracy)


def add_to_writer(writer, epoch, train_loss, valid_loss, train_stats, valid_stats, config):
    writer.add_scalar("Train loss", train_loss, epoch)
    writer.add_scalar("Validation loss", valid_loss, epoch)
    writer.add_scalar("Train f1", train_stats[0], epoch)
    writer.add_scalar("Train accuracy", train_stats[1], epoch)
    writer.add_scalar("Valid f1", valid_stats[0], epoch)
    writer.add_scalar("Valid accuracy", valid_stats[1], epoch)


    writer.flush()
