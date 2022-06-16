import os.path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as trn
from PIL import Image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
folder_dir = "../HARRISON/"
filename = "new_tag_list.txt"


config = {
    "batch_size": 64,
    "epochs": 100,
    "learning_rate": 0.001,
    "window": 4,
    "threshold": 4,
    "embedding_size": 500
}


class Word2Vec(nn.Module):
    def __init__(self, volume_size):
        super(Word2Vec, self).__init__()
        self.embedding_size = config["embedding_size"]
        self.vol = volume_size
        self.u_embeddings = nn.Embedding(self.vol, self.embedding_size)
        self.linear = nn.Linear(self.embedding_size, self.vol)

    def forward(self, x):
        embed_vec = self.u_embeddings(x)
        x = self.linear(embed_vec)
        x = F.log_softmax(x, dim=1)
        return x, embed_vec


class Word2VecDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item][0], self.data[item][1]


def img_centre_crop():
    centre_crop = trn.Compose([
        trn.Resize((256, 256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return centre_crop


def create_vocabulary(file_path, configuration=None):
    corpus = []
    word_counter = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n').strip()
            hashtags = line.strip().split()
            if len(hashtags) > 1:
                corpus.append(hashtags)
                for hashtag in hashtags:
                    if hashtag not in word_counter.keys():
                        word_counter[hashtag] = 1
                    else:
                        word_counter[hashtag] += 1
    return corpus, word_counter, [k for k, v in word_counter.items() if v > configuration["threshold"]]


def convert2pair(data, word2idx):
    idx_data = data.copy()
    for i, row in enumerate(idx_data):
        for i_element, element in enumerate(row):
            if element not in word2idx.keys():
                idx_data[i][i_element] = 0
            else:
                idx_data[i][i_element] = word2idx[element]

    training_set = []
    for i, row in enumerate(idx_data):
        for i_element, element in enumerate(row):
            for slide in range(-config["window"], config["window"] + 1):
                if i_element + slide < 0 or i_element + slide >= len(row) or slide == 0:
                    continue
                else:
                    training_set.append([element, row[i_element + slide]])
    return training_set


class HashtagReader(Dataset):
    def __init__(self, folder_dir, word2idx, word_vec_dict, train=True, one_shot=False):
        super(HashtagReader, self).__init__()
        self.train = train
        self.img_centre_crop = img_centre_crop()
        self.folder_dir = folder_dir

        self.word2idx = word2idx
        self.word_vec_dict = word_vec_dict
        self.one_shot = one_shot

        if train:
            data = pd.read_csv(os.path.join(folder_dir + 'train_data_list.txt'), header=None).values
            tag = pd.read_csv(os.path.join(folder_dir + 'train_tag_list.txt'), header=None).values
        else:
            data = pd.read_csv(os.path.join(folder_dir + 'test_data_list.txt'), header=None).values
            tag = pd.read_csv(os.path.join(folder_dir + 'test_tag_list.txt'), header=None).values

        if one_shot:
            temp = []
            for sentence in tag:
                temp2 = [0] * len(word2idx)
                for hashtag in sentence[0].strip().split(' '):
                    if hashtag in word2idx:
                        temp2[word2idx[hashtag]] = 1
                temp.append(temp2)
            self.shot = temp

        self.y = tag
        self.x = data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = self.x[item]
        y = self.y[item]
        hashtags = y[0].strip()
        category = x[0].split('/')[1]

        category = category[:-1] if category not in self.word2idx.keys() else category
        index = self.word2idx[category]
        hashtag_embeddings = self.word_vec_dict[index]
        temp_path = os.path.join(self.folder_dir, x[0])

        try:
            image = Image.open(temp_path).convert('RGB')
        except Exception:
            temp_path = "/content/HARRISON/instagram_dataset/sun/image_69.jpg"
            image = Image.open(temp_path).convert('RGB')

        input_image = self.img_centre_crop(image)
        if not self.one_shot:
            return input_image, (hashtag_embeddings, hashtags)
        else:
            return input_image, (torch.FloatTensor(self.shot[item]), self.shot[item])


if __name__ == '__main__':
    corpus, word_counter, vocabulary = create_vocabulary(os.path.join(folder_dir, filename), configuration=config)
    word2idx = {k: v for v, k in enumerate(vocabulary, 1)}
    word2idx['UNK'] = 0

    idx2word = {v: k for v, k in enumerate(vocabulary, 1)}
    idx2word[0] = 'UNK'
    vocabulary = vocabulary + ['UNK']

    word_pair = convert2pair(corpus, vocabulary, word2idx)

    training_data = DataLoader(Word2VecDataset(word_pair), batch_size=config["batch_size"], shuffle=True)
    word2vec_model = Word2Vec(len(vocabulary)).to(device)
    optimizer = optim.Adam(word2vec_model.parameters(), lr=config["learning_rate"])
    criterion = nn.NLLLoss()

    for epoch in range(config["epochs"]):
        total_loss = 0
        word2vec_model.train()
        for i, (x, y) in enumerate(training_data, 1):
            x = x.to(device)
            y = y.to(device)
            word2vec_model.zero_grad()
            y_pred, _ = word2vec_model(x)
            loss = criterion(y_pred.view(-1, len(vocabulary)), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print("Epoch {:4d} loss: {:06.4f}".format(epoch, total_loss / len(training_data)))

    with torch.no_grad():
        word2vec_model.eval()
        idx2word = {}
        total = torch.from_numpy(np.arange(len(vocabulary)).reshape(-1, 1)).to(device)
        _, embed_total = word2vec_model(total)
        embed_total = embed_total.reshape((len(vocabulary), -1))
        embed_total_numpy = embed_total.cpu().numpy()
        np.savez('hashtagembed', wordvec=embed_total_numpy)