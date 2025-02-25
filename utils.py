import os
import numpy as np
import cv2
import time
import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from torchvision import transforms as transforms
from datasets import NirVisDataset
import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def CosineSimilarity(a, b):
    a_norm = a / a.norm(dim=1).view(-1, 1)
    b_norm = b / b.norm(dim=1).view(-1, 1)
    score=torch.mm(a_norm, b_norm.t())
    return score

def init_plots_dir():
    global timestr
    timestr = time.strftime("%Y%m%d-%H%M%S")

    global plots_dir
    plots_dir = os.path.join("plots", timestr)
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

def read_list(list_path):
    img_list = []
    with open(list_path, 'r') as f:
        for line in f.readlines()[0:]:
            img_path = line.strip().split()
            img_list.append(img_path[0])
    return img_list

def nir_vis_loader(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        pp = path.split(os.path.sep)
        temp = pp[-1].split('.')
        if temp[-1] == 'bmp':
            temp[-1] = 'jpg'
        elif temp[-1] == 'jpg':
            temp[-1] = 'bmp'
        temp = '.'.join(temp)
        pp[-1] = temp
        i_p = os.path.sep.join(pp)
        img = cv2.imread(i_p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print('image not found')
            print(i_p)
            exit()
    
    return img

def count_succ_recognitions(gallery_features, probe_features, gallery_names, probe_names):
    score = CosineSimilarity(gallery_features,probe_features)
    maxIndex = torch.argmax(score, dim=0)
    count = 0

    for i in range(len(maxIndex)):
        if np.equal(int(probe_names[i]), int(gallery_names[maxIndex[i]])):
            count += 1

    return float(count)

def count_attack_succ_recognitions(gallery_features, probe_features, gallery_names, probe_names):
    score = CosineSimilarity(gallery_features,probe_features)
    maxIndex = torch.argmax(score, axis=0)
    count = 0

    for i in range(len(maxIndex)):
        if not np.equal(int(probe_names[i]), int(gallery_names[maxIndex[i]])):
            count += 1

    return float(count)

def top2_predict(gallery_features, probe_features, gallery_names):
    score = CosineSimilarity(gallery_features, probe_features)
    maxIndex = torch.argmax(score, dim=0)  # Find the index of the maximum score for each probe sample
    top1_predicted_labels = []
    top1_predicted_scores = []
    for i in range(len(maxIndex)):
        top1_predicted_labels.append(int(gallery_names[maxIndex[i]]))
        top1_predicted_scores.append(float(score[maxIndex[i]][i]))
        score[maxIndex[i]][i] = float('-inf')  

    maxIndex = torch.argmax(score, dim=0)
    top2_predicted_labels = []
    top2_predicted_scores = []
    for i in range(len(maxIndex)):
        top2_predicted_labels.append(int(gallery_names[maxIndex[i]]))
        top2_predicted_scores.append(float(score[maxIndex[i]][i]))

    return top1_predicted_labels, top1_predicted_scores, top2_predicted_labels, top2_predicted_scores

def predict(gallery_features, probe_features, gallery_names):
    score = CosineSimilarity(gallery_features, probe_features)
    maxIndex = torch.argmax(score, dim=0)  # Find the index of the maximum score for each probe sample
    predicted_labels = []
    for i in range(len(maxIndex)):
        predicted_labels.append(int(gallery_names[maxIndex[i]]))

    return predicted_labels

def get_label_score(gallery_features, probe_features, gallery_names, target_labels):
    score = CosineSimilarity(gallery_features, probe_features)
    gallery_names = gallery_names.tolist()
    target_labels_indices = [gallery_names.index(target_label) for target_label in target_labels]
    target_scores = []
    for i in range(len(target_labels_indices)):
        target_scores.append(float(score[target_labels_indices[i]][i]))

    return target_scores

# Prepare the paths of the protocol files
def prepare_data_paths(dataset_path, protocols_path, protocol_index):
    gallery_file = 'vis_gallery_' + str(protocol_index) + '.txt'
    probe_file = 'nir_probe_' + str(protocol_index) + '.txt'
    full_protocol_path = os.path.join(dataset_path, protocols_path)
    gallery_file_path = os.path.join(full_protocol_path, gallery_file)
    probe_file_path = os.path.join(full_protocol_path, probe_file)

    if not os.path.exists(gallery_file_path):
        print("Could not found gallery file at", gallery_file_path)

    if not os.path.exists(probe_file_path):
        print("Could not found probe file at", probe_file_path)

    return gallery_file_path, probe_file_path

def save_configuration(args):
    file = open(os.path.join(plots_dir, "config.txt"), "w")
    for arg, value in sorted(vars(args).items()):
        file.write("{}: {}\n".format(arg, value))
    file.close()

def feature_extract(args, images, model):
    images = images.to(device)
    if args.model == "RESNEST":
        images = F.interpolate(images, size=112, mode='bilinear')
        images = images.repeat(1, 3, 1, 1)
        features = model(images)
    else:
        images = F.interpolate(images, size=128, mode='bilinear')
        _, features = model(images)
    return features

def extract_gallery_features(args, model, gallery_file):
    gallery_loader = torch.utils.data.DataLoader(
        NirVisDataset(
            root=args.dataset_path,
            file_list=gallery_file,
            data_type=args.data_type,
            transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    features_dim = 512 if args.model == "RESNEST" else 256
    gallery_size = len(read_list(gallery_file))
    gallery_features = torch.zeros(gallery_size, features_dim).to(device)
    gallery_names = torch.zeros(gallery_size)
    total_time = 0.0
    gallery_dict = {}

    with torch.no_grad():
        for j, (images, labels, _, _) in enumerate(gallery_loader):
            start = time.time()
            features = feature_extract(args, images, model)
            gallery_features[j*args.batch_size:(j+1)*args.batch_size] = features
            gallery_names[j*args.batch_size:(j+1)*args.batch_size] = labels
            for l in labels:
                if l.item() in gallery_dict:
                    msg = f"Duplicated label: {l}, you probably added a subject with an existing label"
                    print(msg)
                    raise ValueError(msg)
            dct = dict(zip([t.item() for t in labels], features))
            gallery_dict.update(dct)

            end = time.time() - start
            total_time += end

    gallery_features = gallery_features.to(device)
    print("Gallery batch extraction duration was {} seconds".format(total_time))

    return gallery_features, gallery_names, gallery_dict

