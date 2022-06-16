import os
import pickle
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as st
from PIL import Image

from source.image_data_reader import create_vocabulary, img_centre_crop
from source.models.resnet_scene import ResnetScene
from source.models.resnet_object import ResnetObject
from source.image_data_reader import HashtagReader
from torch.utils.data import DataLoader
from tqdm import tqdm
from source.evaluation import calc_accuracy, calc_evaluation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = {
    "mode": "analyze",
    "image_path": '/home/ttwicer/Desktop/instagramTagging/HARRISON/instagram_dataset/beach/image_5546.jpg',
    "batch_size": 128,
    "epochs": 100,
    "save": "save_dir/",
    "threshold": 4,
    "learning_rate": 1e-4,
    "continue_training": False,
    "training_name": "objectnscene_vec2word",
    "one_shot": True,
}


def evaluate_hashtag(resnet_object, resnet_scene, folder_dir, word2idx, word_vec_dict):
    valid_dataset = HashtagReader(folder_dir, word2idx, word_vec_dict, train=False, one_shot=config["one_shot"])
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    optimizer_scn = optim.Adam(resnet_scene.parameters(), lr=config["learning_rate"])
    optimizer_obj = optim.Adam(resnet_object.parameters(), lr=config["learning_rate"])
    loss_scn = nn.BCEWithLogitsLoss()
    loss_obj = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        resnet_scene.eval()
        resnet_object.eval()
        running_loss = 0.0
        precision, recall, accuracy, repeat = 0, 0, 0, 0

        val_pbar = tqdm(enumerate(valid_loader, 1), total=len(valid_loader), desc='Validation')
        for j, (val_x, val_y) in val_pbar:
            val_embed_y, val_hashtag_y = val_y
            val_x = val_x.to(device)
            val_embed_y.to(device)

            optimizer_scn.zero_grad()
            optimizer_obj.zero_grad()

            with torch.set_grad_enabled(False):
                val_obj_pred_batch = resnet_object(val_x).detach().cpu()
                val_obj_loss = loss_obj(val_obj_pred_batch, val_embed_y)

                val_scn_pred_batch = resnet_scene(val_x).detach().cpu()
                val_scn_loss = loss_scn(val_scn_pred_batch, val_embed_y)

            running_loss += val_scn_loss.item() + val_obj_loss.item()

            obj_tmp = F.softmax(val_obj_pred_batch, dim=-1)
            scn_tmp = F.softmax(val_scn_pred_batch, dim=-1)

            val_obj_top_k = np.argsort(obj_tmp.detach().cpu().numpy(), axis=1)[:, -5:]
            val_scn_top_k = np.argsort(scn_tmp.detach().cpu().numpy(), axis=1)[:, -5:]

            precision_0, recall_0, accuracy_0 = calc_evaluation(val_obj_top_k, val_embed_y.detach().cpu(), idx2word, one_shot=config["one_shot"], precision_k=1)
            precision_1, recall_1, accuracy_1 = calc_evaluation(val_scn_top_k, val_embed_y.detach().cpu(), idx2word, one_shot=config["one_shot"], precision_k=1)

            precision += (precision_0 + precision_1) / 2
            recall += (recall_0 + recall_1) / 2
            accuracy += (precision_0 + precision_1) / 2
            repeat += 1
    precision /= repeat
    accuracy /= repeat
    recall /= repeat

    with open(os.path.join(config["save"], 'evaluation_results.csv')) as f:
        f.write(f"precision,recall,accuracy\n")
        f.write(f"{precision},{recall},{accuracy}\n")
    print(f"precision,recall,accuracy")
    print(f"{precision},{recall},{accuracy}")


def training(resnet_object, resnet_scene, folder_dir, word2idx, word_vec_dict, vec_matrix):
    train_dataset = HashtagReader(folder_dir, word2idx, word_vec_dict, train=True, one_shot=config["one_shot"])
    valid_dataset = HashtagReader(folder_dir, word2idx, word_vec_dict, train=False, one_shot=config["one_shot"])
    train_loader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    print(f"Validation Data Length: {len(valid_dataset)}\nTrain Data Length: {len(train_dataset)}")
    print()
    total_params = sum(p.numel() for p in resnet_object.parameters()) + sum(
        p.numel() for p in resnet_scene.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in resnet_object.parameters() if p.requires_grad) + sum(
        p.numel() for p in resnet_scene.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    resnet_object.to(device)
    resnet_scene.to(device)
    optimizer_scn = optim.Adam(resnet_scene.parameters(), lr=config["learning_rate"])
    optimizer_obj = optim.Adam(resnet_object.parameters(), lr=config["learning_rate"])
    loss_scn = nn.BCEWithLogitsLoss()
    loss_obj = nn.BCEWithLogitsLoss()

     # writer = SummaryWriter()
    best_train, best_val = 0, 0
    train_info, valid_info = [], []
    for epoch in range(config["epochs"]):
        resnet_scene.train()
        resnet_object.train()

        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc="Epoch {:4d}".format(epoch))
        running_loss = 0.0
        all_tp, all_total, recall = 0, 0, 0
        for i, (x, y) in pbar:
            embed_y, hashtag_y = y

            x = x.to(device)
            embed_y = embed_y.to(device)
            resnet_scene.zero_grad()
            resnet_object.zero_grad()

            obj_pred_batch = resnet_object(x)
            obj_loss = loss_obj(obj_pred_batch, embed_y)

            scn_pred_batch = resnet_scene(x)
            scn_loss = loss_scn(scn_pred_batch, embed_y)

            obj_loss.backward()
            scn_loss.backward()
            optimizer_scn.step()
            optimizer_obj.step()

            running_loss += obj_loss.item() + scn_loss.item()

            obj_tmp = F.softmax(obj_pred_batch, dim=-1)
            scn_tmp = F.softmax(scn_pred_batch, dim=-1)
            obj_top_k = np.argsort(obj_tmp.detach().cpu().numpy(), axis=1)[:, -10:]
            scn_top_k = np.argsort(scn_tmp.detach().cpu().numpy(), axis=1)[:, -10:]

            recall += calc_accuracy(obj_top_k, embed_y.detach().cpu(), idx2word, one_shot=config["one_shot"])
            recall += calc_accuracy(scn_top_k, embed_y.detach().cpu(), idx2word, one_shot=config["one_shot"])
            pbar.set_description("Epoch {:4d}\tLoss {:09.7f}\tRecall {:04.2f}".format(epoch, running_loss / i,
                                                                                      recall / (i * 2) * 100))

        # writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)
        # writer.add_scalar('Accuracy/train', recall, epoch)
        train_info.append([running_loss / len(train_loader), recall])

        if recall > best_train:
            best_train = recall
            tr_path = os.path.join(config["save"], 'best_train')
            torch.save(resnet_object.state_dict(), tr_path + '_resnet_object.pth')
            torch.save(resnet_scene.state_dict(), tr_path + '_resnet_scene.pth')
            print('Weight has been saved for best recall training value.')

        with torch.no_grad():
            resnet_scene.eval()
            resnet_object.eval()
            running_loss = 0.0
            all_tp, all_total, recall = 0, 0, 0

            val_pbar = tqdm(enumerate(valid_loader, 1), total=len(valid_loader), desc='Validation')
            for j, (val_x, val_y) in val_pbar:
                val_embed_y, val_hashtag_y = val_y
                val_x = val_x.to(device)
                val_embed_y.to(device)

                optimizer_scn.zero_grad()
                optimizer_obj.zero_grad()

                with torch.set_grad_enabled(False):
                    val_obj_pred_batch = resnet_object(val_x).detach().cpu()
                    val_obj_loss = loss_obj(val_obj_pred_batch, val_embed_y)

                    val_scn_pred_batch = resnet_scene(val_x).detach().cpu()
                    val_scn_loss = loss_scn(val_scn_pred_batch, val_embed_y)

                running_loss += val_scn_loss.item() + val_obj_loss.item()

                obj_tmp = F.softmax(val_obj_pred_batch, dim=-1)
                scn_tmp = F.softmax(val_scn_pred_batch, dim=-1)

                val_obj_top_k = np.argsort(obj_tmp.detach().cpu().numpy(), axis=1)[:, -5:]
                val_scn_top_k = np.argsort(scn_tmp.detach().cpu().numpy(), axis=1)[:, -5:]

                recall += calc_accuracy(val_obj_top_k, val_embed_y.detach().cpu(), idx2word,
                                        one_shot=config["one_shot"])
                recall += calc_accuracy(val_scn_top_k, val_embed_y.detach().cpu(), idx2word,
                                        one_shot=config["one_shot"])
                val_pbar.set_description("Validation Epoch {:4d}\tLoss {:09.7f}\tRecall {:04.2f}"
                                         .format(epoch, running_loss / j, recall / (j * 2) * 100))

            if recall > best_val:
                best_val = recall
                val_path = os.path.join(config["save"], 'best_val')
                torch.save(resnet_object.state_dict(), val_path + '_resnet_object.pth')
                torch.save(resnet_scene.state_dict(), val_path + '_resnet_scene.pth')
                print('Weight has been saved for best recall validation value.')

            # writer.add_scalar('Loss/validation', running_loss / len(valid_loader), epoch)
            # writer.add_scalar('Accuracy/validation', recall, epoch)
            train_info.append([running_loss / len(valid_loader), recall])

    print('Training phase successfully ended.')
    print('Writing overall train/validation metrics.')
    with open(os.path.join(config["save"], 'train_results.csv')) as f:
        for loss, recall in train_info:
            f.write(f"{loss},{recall}\n")

    with open(os.path.join(config["save"], 'valid_results.csv')) as f:
        for loss, recall in valid_info:
            f.write(f"{loss},{recall}\n")
    # writer.close()


def predict(resnet_object, resnet_scene, idx2word, pickles, configuration):
    image = Image.open(configuration["image_path"]).convert('RGB')
    centre = img_centre_crop()
    input_image = centre(image).unsqueeze(0)

    resnet_scene.eval()
    resnet_object.eval()

    with torch.no_grad():
        obj_pred = resnet_object(input_image)
        scn_pred = resnet_scene(input_image)
        results = []
        obj_tmp = F.softmax(obj_pred, dim=-1)
        scn_tmp = F.softmax(scn_pred, dim=-1)

        obj_conf, obj_classes = torch.topk(obj_tmp, 20, 1)
        scn_conf, scn_classes = torch.topk(scn_tmp, 20, 1)
        obj_conf = obj_conf.detach().cpu()[0].numpy()
        obj_classes = obj_classes.detach().cpu()[0].numpy()
        scn_conf = scn_conf.detach().cpu()[0].numpy()
        scn_classes = scn_classes.detach().cpu()[0].numpy()
        conf_temp = []
        class_temp = []

        for i in range(len(obj_classes)):
            if obj_classes[i] not in class_temp:
                conf_temp.append(obj_conf[i])
                class_temp.append(obj_classes[i])
            else:
                index = np.where(class_temp == obj_classes[i])
                conf_temp[index[0][0]] += obj_conf[i]
            if scn_classes[i] not in class_temp:
                conf_temp.append(scn_conf[i])
                class_temp.append(scn_classes[i])
            else:
                index = np.where(class_temp == scn_classes[i])
                conf_temp[index[0][0]] += scn_conf[i]

        for i in range(len(conf_temp)):
            temp_mean = pickles[0][class_temp[i]]
            temp_std = pickles[1][class_temp[i]]
            results.append(('#' + idx2word[class_temp[i]], st.norm.cdf((conf_temp[i] - temp_mean) / temp_std) * 100))
            # results.append(('#' + idx2word[class_temp[i]], conf_temp[i] * 20 / temp_mean))
        dtypes = [('hashtag', 'U25'), ('percentage', float)]
        predicts = np.array(results, dtype=dtypes)
        predicts = np.sort(predicts, order='percentage')
        predicts = np.flip(predicts)
        return predicts[:10]


def analyze_hashtags(resnet_object, resnet_scene, idx2word, configuration):
    valid_dataset = HashtagReader(folder_dir, word2idx, word_vec_dict, train=False, one_shot=config["one_shot"])
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    resnet_object.eval()
    resnet_scene.eval()
    pbar = tqdm(enumerate(valid_loader, 1), total=len(valid_loader), desc='Validation')

    confidences = {i: 0 for i in range(len(idx2word))}
    repeat = {i: 0 for i in range(len(idx2word))}
    standard = {i: 0 for i in range(len(idx2word))}
    max_value = {i: 0 for i in range(len(idx2word))}
    min_value = {i: 0 for i in range(len(idx2word))}
    samples = {i: [] for i in range(len(idx2word))}
    for i, (x, _) in pbar:
        obj_pred_batch = resnet_object(x).detach().cpu()
        scn_pred_batch = resnet_scene(x).detach().cpu()
        obj_tmp = F.softmax(obj_pred_batch, dim=-1)
        scn_tmp = F.softmax(scn_pred_batch, dim=-1)

        obj_conf, obj_classes = torch.topk(obj_tmp, 10, 1)
        scn_conf, scn_classes = torch.topk(scn_tmp, 10, 1)

        obj_conf = obj_conf.detach().cpu()[0].numpy()
        obj_classes = obj_classes.detach().cpu()[0].numpy()
        scn_conf = scn_conf.detach().cpu()[0].numpy()
        scn_classes = scn_classes.detach().cpu()[0].numpy()

        for j in range(len(obj_classes)):
            confidences[obj_classes[j]] += obj_conf[j]
            samples[obj_classes[j]].append(obj_conf[j])
            repeat[obj_classes[j]] += 1

        for j in range(len(scn_classes)):
            confidences[scn_classes[j]] += scn_conf[j]
            samples[obj_classes[j]].append(obj_conf[j])
            repeat[scn_classes[j]] += 1

    for i in range(len(confidences)):
        if repeat[i] != 0:
            confidences[i] /= repeat[i]
        print(f"{idx2word[i]}={confidences[i]}")

    for i in range(len(standard)):
        if len(samples[i]) > 0:
            temp = np.array(samples[i])
            max_value[i] = np.max(temp)
            min_value[i] = np.min(temp)
            standard[i] = np.std(temp)
        else:
            standard[i] = 0
            max_value[i] = 0
            min_value[i] = 0

    with open(os.path.join(config["save"], 'saved_mean_10_v2.pkl'), 'wb') as f:
        pickle.dump(confidences, f)

    with open(os.path.join(config["save"], 'saved_repeat_10_v2.pkl'), 'wb') as f:
        pickle.dump(repeat, f)

    with open(os.path.join(config["save"], 'saved_standard_10_v2.pkl'), 'wb') as f:
        pickle.dump(standard, f)

    with open(os.path.join(config["save"], 'saved_max_10_v2.pkl'), 'wb') as f:
        pickle.dump(max_value, f)

    with open(os.path.join(config["save"], 'saved_min_10_v2.pkl'), 'wb') as f:
        pickle.dump(min_value, f)


if __name__ == '__main__':
    # predict()
    folder_dir = '/content/HARRISON/'
    filename = 'new_tag_list.txt'

    vec_matrix = np.load("hashtagembed.npz")['wordvec']
    word_vec_dict = {idx: vec for idx, vec in enumerate(vec_matrix)}
    _, _, vocabulary = create_vocabulary(os.path.join(folder_dir, filename), config)
    word2idx = {k: v for v, k in enumerate(vocabulary, 1)}
    word2idx['UNK'] = 0

    idx2word = {v: k for v, k in enumerate(vocabulary, 1)}
    idx2word[0] = 'UNK'
    vocabulary = vocabulary + ['UNK']

    resnet_object = ResnetObject(len(vocabulary) if config["one_shot"] else 500)
    resnet_scene = ResnetScene(len(vocabulary) if config["one_shot"] else 500)

    if config["mode"] == "train":
        if config["continue_training"]:
            path = os.path.join(config["save"], 'best_val')

        if not os.path.exists(config["save"]):
            os.makedirs(config["save"])

        try:
            training(resnet_object, resnet_scene, folder_dir, word2idx, word_vec_dict, vec_matrix)
        except KeyboardInterrupt:
            print('The training has been manually cancelled.')
            try:
                sys.exit()
            except SystemExit:
                os._exit(1)
    elif config["mode"] == "analyze":
        path = os.path.join(config["save"], 'best_val')
        resnet_object.load_state_dict(torch.load(config["save"] + 'best_val_resnet_object.pth',
                                                 map_location=torch.device('cpu')))
        resnet_scene.load_state_dict(torch.load(config["save"] + 'best_val_resnet_scene.pth',
                                                map_location=torch.device('cpu')))
        analyze_hashtags(resnet_object, resnet_scene, idx2word, config)
    elif config["mode"] == "evaluate":
        path = os.path.join(config["save"], 'best_val')
        resnet_object.load_state_dict(torch.load(config["save"] + 'best_val_resnet_object.pth',
                                                 map_location=torch.device('cpu')))
        resnet_scene.load_state_dict(torch.load(config["save"] + 'best_val_resnet_scene.pth',
                                                map_location=torch.device('cpu')))
        evaluate_hashtag(resnet_object, resnet_scene, folder_dir, word2idx, word_vec_dict)

    else:
        raise ValueError(f'config mode specification could not recognized: {config["mode"]}')
