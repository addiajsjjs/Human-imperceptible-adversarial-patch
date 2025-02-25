import utils
import torch
import torchvision.transforms as transforms
from datasets import NirVisDataset, NirVisDatasetPerSubject
import argparse
import os 
import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def before_predict(args, model, gallery_file_path, probe_file_path):
    # Extracting features from gallery images
    (gallery_features, gallery_names, gallery_dict) = utils.extract_gallery_features(args, model, gallery_file_path)

    # Preparing the dataset and DataLoader based on the type of attack (physical or digital)
    if args.physical_attack:
        probe_dataset = NirVisDatasetPerSubject(
            root=args.dataset_path,
            file_list=probe_file_path,
            transform=transforms.Compose([transforms.ToTensor()]))
        batch_size = None  # Process all images of the attacker
        _, true_labels, target_labels, _ = probe_dataset[0]
    else:
        probe_dataset = NirVisDataset(
            root=args.dataset_path,
            file_list=probe_file_path,
            transform=transforms.Compose([transforms.ToTensor()]))
        batch_size = args.batch_size
   
    probe_loader = torch.utils.data.DataLoader(probe_dataset, batch_size=batch_size, num_workers=args.workers, pin_memory=True)

    for j, (images, true_labels, target_labels, _) in enumerate(probe_loader):
        images = images.to(device)

        with torch.no_grad():
            probe_features = utils.feature_extract(args, images, model)

        predicted_labels = utils.predict(gallery_features, probe_features, gallery_names)
        print(f"The following subjects:{true_labels.tolist()} were predicted as {predicted_labels}")

def predict(args, model, gallery_file_path, probe_file_path):
    # Extracting features from gallery images
    (gallery_features, gallery_names, gallery_dict) = utils.extract_gallery_features(args, model, gallery_file_path)
    
    probe_dataset = NirVisDataset(
            root=args.dataset_path,
            file_list=probe_file_path,
            data_type=args.data_type,
            transform=transforms.Compose([transforms.ToTensor()]))
    batch_size = args.batch_size

    probe_loader = torch.utils.data.DataLoader(probe_dataset, batch_size=batch_size, num_workers=args.workers, pin_memory=True)
    check = 0
    total_num_of_examples = 0
    for j, (images, true_labels, target_labels, _) in enumerate(probe_loader):
        with torch.no_grad():
            probe_features = utils.feature_extract(args, images, model)

        top1_predicted_labels, top1_predicted_scores, top2_predicted_labels, top2_predicted_scores = utils.top2_predict(gallery_features, probe_features, gallery_names)
        target_labels = top2_predicted_labels
        

        after_target_label_scores = utils.get_label_score(gallery_features, probe_features, gallery_names, target_labels)
        after_true_label_scores = utils.get_label_score(gallery_features, probe_features, gallery_names, true_labels)
        check_suu = utils.count_succ_recognitions(gallery_features, probe_features, gallery_names, true_labels)
        check += check_suu
        num_of_examples = len(true_labels)
        total_num_of_examples += num_of_examples
        print(f"The following subjects:{true_labels.tolist()} were predicted as {top1_predicted_labels}")
        print(top1_predicted_scores)
        print(after_true_label_scores)
        print(after_target_label_scores)
        #print(f"{top1_predicted_scores}")
        #print(top1_predicted_scores)
        #print(f"The following subjects:{predicted_label} were predicted as {predicted_labels2}")        
    print(check)
    print(total_num_of_examples)
    print(f"acc:{check/total_num_of_examples}")

if __name__ == "__main__":
    gallery_file_path = ''
    probe_file_path = ''
    parser = argparse.ArgumentParser(description='Process some dataset.')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset')
    parser.add_argument('--gallery_file_path', type=str, default=128)
    parser.add_argument('--probe_file_path', type=str, default=128)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--model', type=str, default='DVG')
    parser.add_argument('--pretrained_path',  type=str)
    parser.add_argument('--data_type', default='caisa', type=str,choices=["casia","buua","oulu"])
    args = parser.parse_args()
    model = load_model.load_model(args)
    predict(args, model, args.gallery_file_path, args.probe_file_path)