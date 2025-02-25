import os
import random
import kornia
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
import torch
from PIL import Image
from untils import mapping3d,rotate,stick,feature
from models import image_transform_layers
import cv2
from scipy.ndimage import gaussian_filter
from utils import count_succ_recognitions, mask_color_init, feature_extract, count_attack_succ_recognitions
import utils
import math
from scipy.interpolate import splprep, splev
from scipy.spatial import distance
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DEIndividal:
    def __init__(self, centers=None, AreaNum=None, lenthes=None):
        self.fitness = 1.0
        self.AreaNum = AreaNum
        self.centers = centers
        self.lenthes = lenthes

    def generate_attack_mask_simple(self, image, mask):
        angle = 360 / self.AreaNum
        angle = math.radians(angle)
        attack_image = image.copy()
        attack_mask = np.zeros_like(image)

        image_np = np.array(image)
        image_np = image_np.astype(float)
        #LRM strategy
        ink_image_np = image_np * 0.333

        
        for center, lenth in zip(self.centers, self.lenthes):
            points = []
            y,x = center

            for i in range(self.AreaNum):
                new_x = x + lenth[i] * math.cos(i * angle)
                new_y = y + lenth[i] * math.sin(i * angle)
                points.append([new_x, new_y])

            points.append(points[0])  
            points = np.array(points)

            tck, u = splprep([points[:, 0], points[:, 1]], s=0, per=True)
            u_fine = np.linspace(0, 1, num=1000)
            spline_points = splev(u_fine, tck)
            spline_points = np.array(spline_points).T

            
            spline_points = np.rint(spline_points).astype(np.int32)
            height, width = attack_mask.shape[:2]
            spline_points[:, 0] = np.clip(spline_points[:, 0], 0, width - 1)
            spline_points[:, 1] = np.clip(spline_points[:, 1], 0, height - 1)
            spline_points = spline_points.reshape((-1, 1, 2))
            cv2.fillPoly(attack_mask, [spline_points], 255)

            
            attack_image[attack_mask == 255] = ink_image_np[attack_mask == 255]
            
        attack_image[mask == 0] = image[mask==0]

        return attack_image

    def generate_attack_mask_simple1(self, image, mask):
        angle = 360 / self.AreaNum
        angle = math.radians(angle)
        attack_image = image.copy()
        attack_mask = np.zeros_like(image)
        attack_mask_tmp = np.zeros_like(image)
        image_np = np.array(image)
        image_np = image_np.astype(float)
        ink_image_np = image_np * 0.333

        
        for center, lenth in zip(self.centers, self.lenthes):
            points = []
            y,x = center

            for i in range(self.AreaNum):
                new_x = x + lenth[i] * math.cos(i * angle)
                new_y = y + lenth[i] * math.sin(i * angle)
                points.append([new_x, new_y])

            points.append(points[0])  
            points = np.array(points)

            tck, u = splprep([points[:, 0], points[:, 1]], s=0, per=True)
            u_fine = np.linspace(0, 1, num=1000)
            spline_points = splev(u_fine, tck)
            spline_points = np.array(spline_points).T

            
            spline_points = np.rint(spline_points).astype(np.int32)
            height, width = attack_mask.shape[:2]
            spline_points[:, 0] = np.clip(spline_points[:, 0], 0, width - 1)
            spline_points[:, 1] = np.clip(spline_points[:, 1], 0, height - 1)
            spline_points = spline_points.reshape((-1, 1, 2))
            cv2.fillPoly(attack_mask, [spline_points], 255)

            
            attack_image[attack_mask == 255] = ink_image_np[attack_mask == 255]
        attack_image[mask == 0] = image[mask==0]
        attack_mask_tmp[attack_mask == 0] = 255
        attack_mask_tmp[attack_mask == 255] = 0
        return attack_mask_tmp

    def generate_attack_mask(self, image, mask):
        angle = 360 / self.AreaNum
        angle = math.radians(angle)
        attack_image = image.copy()
        attack_mask = np.zeros_like(image)
        image_np = np.array(image)
        image_np = image_np.astype(float)
        ink_image_np = image_np * 0.333
       
        for center, lenth in zip(self.centers, self.lenthes):
            points = []
            y,x = center

            for i in range(self.AreaNum):
                new_x = x + lenth[i] * math.cos(i * angle)
                new_y = y + lenth[i] * math.sin(i * angle)
                points.append([new_x, new_y])

            points.append(points[0])  
            points = np.array(points)

            tck, u = splprep([points[:, 0], points[:, 1]], s=0, per=True)
            u_fine = np.linspace(0, 1, num=1000)
            spline_points = splev(u_fine, tck)
            spline_points = np.array(spline_points).T

            
            spline_points = np.rint(spline_points).astype(np.int32)
            height, width = attack_mask.shape[:2]
            spline_points[:, 0] = np.clip(spline_points[:, 0], 0, width - 1)
            spline_points[:, 1] = np.clip(spline_points[:, 1], 0, height - 1)
            spline_points = spline_points.reshape((-1, 1, 2))
            cv2.fillPoly(attack_mask, [spline_points], 255)


            attack_image[attack_mask == 255] = 0
        attack_image[mask == 0] = image[mask==0]

        
        attack_image = PIL2torch(attack_image)

        return attack_image


    

class Deformable_ink_attack:
    
    def __init__(self, args, model, gallery_features, true_names, target_names, gallery_names, maxiter, sizepop, AreaNum, AnchorNum):
        self.args = args
        self.model = model
        self.gallery_features = gallery_features
        self.true_names = true_names
        self.target_names = target_names
        self.gallery_names = gallery_names
        self.maxiter = maxiter
        self.sizepop = sizepop
        self.AreaNum = AreaNum
        self.AnchorNum = AnchorNum


    def generate_face_mask(self, image):
        mask = feature.make_mask(image)
        return mask
        
    def visualize_score(self, score, save_path=None):
            score[score == -np.inf] = 0
            norm_score = (score - score.min()) / (score.max() - score.min())
            norm_score = (norm_score * 255).astype(np.uint8)

            if save_path:
                cv2.imwrite(save_path, norm_score)

    def initialize(self, mask):
        DE_all = []
        coords = np.argwhere(mask==1)
        for _ in range(self.sizepop):
            lenthes = []
            centers = []
            for _ in range(self.AnchorNum):
                centers.append(coords[np.random.choice(coords.shape[0])])
                lenthes.append([random.randint(2, 20) for _ in range(self.AreaNum)])
            DE_all.append(DEIndividal(centers, self.AreaNum, lenthes))
        return DE_all


    def MutationOperation(self, DE_all, image, mask, CR=0.3):

        DE_new = []
        h,w = image.shape
        for i in range(self.sizepop):
            lenthes = DE_all[i].lenthes.copy()  
            centers = DE_all[i].centers.copy()

            a = np.random.randint(0, self.sizepop - 1)
            while a == i:
                a = np.random.randint(0, self.sizepop - 1)

            b = np.random.randint(0, self.sizepop - 1)
            while b == i or b == a:
                b = np.random.randint(0, self.sizepop - 1)


            a_lenthes = DE_all[a].lenthes
            b_lenthes = DE_all[b].lenthes
            a_centers = DE_all[a].centers
            b_centers = DE_all[b].centers

            NEWcenters = []
            for center, a_center, b_center in zip(centers, a_centers, b_centers):
                center[0] += int(center[0] + CR * (a_center[0]-b_center[0]))
                center[1] += int(center[1] + CR * (a_center[1]-b_center[1]))
                center[0] = max(0, min(center[0], h-1))
                center[1] = max(0, min(center[1], w-1))
                if mask[center[0]][center[1]] == 1:
                    NEWcenters.append([center[0],center[1]])
                else:
                    coords = np.argwhere(mask==1)
                    NEWcenters.append(coords[np.random.choice(coords.shape[0])])

            NEWlenthes = []
            for lenth, a_lenth, b_lenth in zip(lenthes, a_lenthes, b_lenthes):
                new_lenth = []
                for j in range(self.AreaNum):
                    value = lenth[j] + CR * (a_lenth[j] - b_lenth[j])
                    value = max(2, min(value, 20))
                    new_lenth.append(value)
                NEWlenthes.append(new_lenth)

            new_individual = DE_all[i].__class__()
            new_individual.AreaNum = self.AreaNum
            new_individual.lenthes = NEWlenthes
            new_individual.centers = NEWcenters


            DE_new.append(new_individual)

        return DE_new
            
    def CrossoverOperation(self, DE_all, DE_mutation, CR=0.9):

        DE_new = []
        for i in range(self.sizepop):
            original_lenthes = DE_all[i].lenthes
            mutation_lenthes = DE_mutation[i].lenthes
            original_centers = DE_all[i].centers
            mutation_centers = DE_mutation[i].centers         
            new_lenthes = []
            #print(f'original:{original_lenthes}')
            #print(f'mutation:{mutation_lenthes}')
            for orig_lenth, mut_lenth in zip(original_lenthes, mutation_lenthes):
                crossover_lenth = []
                for j in range(self.AreaNum):
                    rand_val = random.random()
                    #print(f'rand_val:{rand_val}')
                    if rand_val < CR or j == np.random.randint(0, self.AreaNum):
                        crossover_lenth.append(mut_lenth[j])
                    else:
                        crossover_lenth.append(orig_lenth[j])
                crossover_lenth = [max(2, min(l, 20)) for l in crossover_lenth]
                new_lenthes.append(crossover_lenth)
            #print(f'new:{new_lenthes}')

            crossover_centers = []
            for orig_center, mut_center in zip(original_centers, mutation_centers):
                
                rand_val = random.random()
                if rand_val < CR:
                    crossover_centers.append(mut_center)
                else:
                    crossover_centers.append(orig_center)

            new_individual = DE_all[i].__class__()
            new_individual.AreaNum = self.AreaNum
            new_individual.lenthes = new_lenthes
            new_individual.centers = crossover_centers


            DE_new.append(new_individual)    

        return DE_new

    def CaculateFitness(self, DE_all, image, true_name, mask):
        attack_images = []
        for DE_one in DE_all:
            attack_image = DE_one.generate_attack_mask(image, mask)
            attack_images.append(attack_image)
        attack_images_tensor = torch.stack(attack_images)
        perturbed_features = feature_extract(self.args, attack_images_tensor, self.model)
        with torch.no_grad():
            top1_labels, top1_scores, top2_labels, top2_scores = utils.top2_predict(self.gallery_features, 
                                                                              perturbed_features, self.gallery_names)
        for DE_one, top1_label, top1_score, top2_label, top2_score in zip(DE_all, top1_labels, top1_scores, top2_labels, top2_scores):
            if top1_label != true_name:
                DE_one.fitness = 0
            else:
                DE_one.fitness = top1_score - top2_score
        return DE_all, top1_labels
        
    def attack(self, DE_all, image, true_name, mask):
        t = 0
        while(t < self.maxiter):
            #print(f't:{t}')
            DE_mutation = self.MutationOperation(DE_all,image,mask)
            DE_next = self.CrossoverOperation(DE_all, DE_mutation)
            DE_next, top1_labels_next = self.CaculateFitness(DE_next, image, true_name, mask)
            DE_all, top1_labels_all = self.CaculateFitness(DE_all, image, true_name, mask)
            top1_labels = []
            for i in range(len(DE_all)):
                #print(f'DE_next[i].fitness:{DE_next[i].fitness}')
                #print(f'DE_all[i]:{DE_all[i].fitness}')
                if DE_next[i].fitness < DE_all[i].fitness:
                    DE_all[i].centers = DE_next[i].centers
                    DE_all[i].lenthes = DE_next[i].lenthes
                    DE_all[i].fitness = DE_next[i].fitness
                    #print('change!')
                    top1_labels.append(top1_labels_next[i])
                else:
                    top1_labels.append(top1_labels_all[i])
                    #print('unchange!')
                    continue

            min_fitness_individual = min(DE_all, key=lambda x: x.fitness)

            if not all(label == true_name for label in top1_labels):
                return DE_all, top1_labels, True   
            t+=1
        return DE_all, top1_labels, False


    def excute(self, args, images):
        check = 0
        for image, true_name in zip(images, self.true_names):
            print(f'solve untargeted attack:{true_name}')
            image = torch2uint8gray(image)
            mask = self.generate_face_mask(image)
            DE_all = self.initialize(mask)
            DE_all, top1_labels, attack_succ = self.attack(DE_all, image, true_name, mask)
            if attack_succ:
                print('attack_succ! BUT DEF!')
                first_different_label = next((label for label in top1_labels if label != true_name), None)
                first_different_label_index = top1_labels.index(first_different_label)
                output_path = f'DEF-attack-{true_name}-{first_different_label}-anchor{self.AnchorNum}-area{self.AreaNum}.jpg'
                output_dir = f'{args.model}-{args.data_type}-{args.attack_type}-score-images'
                full_output_path = os.path.join(output_dir, output_path)
                attack_succ_image = DE_all[first_different_label_index].generate_attack_mask_simple(image,mask)
                attack_succ_mask = DE_all[first_different_label_index].generate_attack_mask_simple1(image,mask)
                cv2.imwrite(full_output_path, attack_succ_image)
                #cv2.imwrite(f"{args.model}-{args.data_type}-{args.attack_type}-{true_name}-{first_different_label}.jpg",attack_succ_mask)            
                continue

            else:
                print('attack_fail!')
                check+=1
             
        return check




def torch2uint8gray(img):
        if img.dim() == 3 and img.size(0) == 1:
            img = img.squeeze(0)
        
        image_np = img.cpu().numpy() 
        image = (image_np * 255).astype(np.uint8)
        return image 

def PIL2torch(img):
        transform = transforms.ToTensor()
        outImage_tensor = transform(img)
        if outImage_tensor.shape[0] == 1 and len(outImage_tensor.shape) == 3:
            outImage_tensor = outImage_tensor 
        else:
            outImage_tensor = outImage_tensor.squeeze(0)
            outImage_tensor = torch.unsqueeze(outImage_tensor, 0)
        
        return outImage_tensor
