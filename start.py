import argparse
import torch.utils.data as data
import os
import os.path
import torch
from itertools import groupby
import random
import load_model
import utils
from datasets import NirVisDataset
import torchvision.transforms as transforms
import time
from Deformable_attack import Deformable_ink_attack
from Deformable_attack_target import Deformable_ink_attack_target
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#python attack_single.py
def perform_attack(args, model, gallery_file_path, probe_file_path):
  start = time.time()
  
  total_num_of_examples, running_targeted_att_succ_rate, running_before_att_succ_rate, running_after_att_succ_rate = 0, 0, 0, 0

  (gallery_features, gallery_names, gallery_dict) = utils.extract_gallery_features(args, model, gallery_file_path)
  
  probe_dataset = NirVisDataset(
    root=args.dataset_path,
    file_list=probe_file_path,
    data_type=args.data_type,
    transform=transforms.Compose([transforms.ToTensor()])
  )
  batch_size = args.batch_size
  num_of_images_to_probe = len(probe_dataset)
  

  probe_loader = torch.utils.data.DataLoader(probe_dataset,batch_size=batch_size,num_workers=args.workers,pin_memory=True)

  for j,(images, true_labels, target_labels, pose_transformation_matrices) in enumerate(probe_loader):
    
    with torch.no_grad():
      probe_features = utils.feature_extract(args, images, model)

      succ_recognitions_before_attack = utils.count_succ_recognitions(gallery_features,probe_features,gallery_names,true_labels)
      top1_labels, top1_scores, top2_labels, top2_scores = utils.top2_predict(gallery_features, 
                                                                              probe_features, gallery_names)
      if args.attack_type == 'untargeted_attack':
        maxiter = 400
        sizepop = 40
        ink_DE_vor_attack = Deformable_ink_attack(args, model, gallery_features, true_labels, target_labels, gallery_names,  maxiter, sizepop, args.AreaNum, args.AnchorNum)

        succ_recognitions_after_attack = ink_DE_vor_attack.excute(args,images)
      elif args.attack_type == 'targeted_attack':
        sizepop = 40
        top1_predicted_labels, top1_predicted_scores, top2_predicted_labels, top2_predicted_scores = utils.top2_predict(gallery_features, probe_features, gallery_names)
        target_labels = top2_predicted_labels
        ink_DE_vor_attack = Deformable_ink_attack_target(args, model, gallery_features, true_labels, target_labels, gallery_names,  args.maxiter, sizepop, args.AreaNum, args.AnchorNum)
        succ_recognitions_after_attack = ink_DE_vor_attack.excute(args,images)
      else:
        print("Error - attack type is wrong")
        exit()
  
       # Updating performance metrics
      num_of_examples = len(true_labels)
      total_num_of_examples += num_of_examples
      running_after_att_succ_rate += succ_recognitions_after_attack
      running_before_att_succ_rate += succ_recognitions_before_attack

  print(f'running_before:{running_before_att_succ_rate}')
  print(f'running_after:{running_after_att_succ_rate}')
  end = time.time() - start

  print(f"Attack duration was {end} seconds")
  print(f'model:{args.model}')
  print(f'running_before_att_succ_rate:{running_before_att_succ_rate}')
  print(f'running_after_att_succ_rate:{running_after_att_succ_rate}')
  if args.attack_type == 'untargeted_attack':    
    print(f'ASR:{(running_before_att_succ_rate-running_after_att_succ_rate)/running_before_att_succ_rate}')
  if args.attack_type == 'targeted_attack':
    print(f'ASR:{(running_after_att_succ_rate)/total_num_of_examples}')
  return running_before_att_succ_rate, running_after_att_succ_rate, running_targeted_att_succ_rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some dataset.')
    parser.add_argument('--gallery_file_path', type=str)
    parser.add_argument('--probe_file_path', type=str)
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset')
    parser.add_argument('--attack_type', type=str, default='untargeted_attack')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--model', type=str, default='RESNEST')
    parser.add_argument('--pretrained_path', type=str)
    parser.add_argument('--maxiter', default=200, type=int)
    parser.add_argument('--data_type', default='casia', type=str,choices=["casia","buua","oulu"])
    parser.add_argument('--AreaNum', default='8', type=int)
    parser.add_argument('--AnchorNum', default='4', type=int)
    args = parser.parse_args()
    model = load_model.load_model(args)
    perform_attack(args, model, args.gallery_file_path, args.probe_file_path)
