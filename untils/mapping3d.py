import os, sys
import subprocess
import numpy as np
from numpy.linalg import *
import dlib
import cv2
import scipy.io as sio
from skimage import io
from time import time
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as imgplt
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
from scipy.optimize import fsolve
import math
from torchvision import datasets
from untils import stick

#sys.path.append('..')
import face3d
from face3d import mesh
from face3d.morphable_model import MorphabelModel

def landmarks_68(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    pic_array = np.array(img)
    h, w = pic_array.shape
    #r,g,b = cv2.split(pic_array)
    #img = cv2.merge([b, g, r])
    #img_gray = cv2.cvtColor(pic_array, cv2.COLOR_RGB2GRAY)   
    rects = detector(pic_array, 1)                      
    #print(len(rects))
    # rawpots = np.array([[p.x, p.y] for p in predictor(img,rects[0]).parts()])
    if len(rects) == 0:
        raise ValueError('No faces detected in the image')

    feature_points = np.array([[(p.x-w/2), (h/2-p.y)] for p in predictor(img,rects[0]).parts()])
    #print(feature_points)
    return feature_points

def sticker_spatial(vertices, img):
    w, h = img.shape
    store = np.zeros((h,w,2))
    for i in range(len(vertices)):
        x = int(np.round(vertices[i][0]))
        y = int(np.round(vertices[i][1]))
        x = min(x,w-1)
        y = min(y,h-1)
        store[y][x][0] = store[y][x][0] + vertices[i][2]
        store[y][x][1] = store[y][x][1] + 1
    store[:,:,1][np.where(store[:,:,1]==0)] = 1
    zstore = store[:,:,0]/store[:,:,1]

    return zstore

def generate_zstore(img):
    # --- 1. load model
    bfm = MorphabelModel('./BFM/BFM.mat')
    #print('init bfm model success')

    # --- 2. load fitted face
    feature_points = landmarks_68(img)
    w,h = img.shape

    x = feature_points.copy()
    X_ind = bfm.kpt_ind # index of keypoints in 3DMM. fixed.
    sp, ep, s, angles, t3d = bfm.fit(x, X_ind, max_iter = 15)
    #print('s, angles, t = ',s, angles, t3d)

    # tp = bfm.get_tex_para('random')
    # colors1 = bfm.generate_colors(tp)

    # verify fitted parameters
    vertices = bfm.generate_vertices(sp, ep)
    transformed_vertices = bfm.transform(vertices, s, angles, t3d)

    image_vertices = mesh.transform.to_image(transformed_vertices, h, w)
    zstore = sticker_spatial(image_vertices, img)
    return zstore

def comp_arclen(a,c,x):
    def f(x):
        return (1 + ((x-c)*(2*a))**2)**0.5
    A = integrate.quad(f,0,x)[0]
    return A

def binary_equation(y1,z1,y2,z2):
    def flinear(x):# k:x[0], b:x[1]
        return np.array([x[0]*y1+x[1]-z1,x[0]*y2+x[1]-z2])
    yzlinear = fsolve(flinear,[0,0])
    return yzlinear

def solve_b(a,c,w,locate):
    '''
    find the upper limit of the integral 
    so that the arc length is equal to the width of the sticker
    locate: the highest point is on th right(1) b = -a*c^2, 
                                     left (2) b = -a*(upper-c)^2
    return: b , wn=upper(Width of converted picture)
    '''
    def func(x):
        def f(x):
            return (1 + ((x-c)*(2*a))**2)**0.5
        f = integrate.quad(f,0,x)[0]-w
        return f
    root = 0
    upper = fsolve(func,[root])[0] # The X coordinate when the arc length = w
    if(locate == 1):
        b = -a * (c**2)
    elif(locate == 2):
        b = -a * ((upper-c)**2)
    wn = int(np.floor(upper))
    return b, wn

def solve_a(hsegment,stride):
    '''
    solve 'a' according to Height drop in one step
    '''
    a = -hsegment/(stride)**2
    a = max(a, -1/stride)
    #print('a=',a,'hsegment=',hsegment,'stride=',stride)
    return a

def bilinear_interpolation(img,x,y):
    w,h = img.shape
    xset = math.modf(x)
    yset = math.modf(y)
    u, v = xset[0], yset[0]
    x1, y1 = int(xset[1]), int(yset[1])
    x2 = x1+1 if u>0 and x1+1<w else x1
    y2 = y1+1 if v>0 and y1+1<h else y1
    #x2, y2 = x1+1, y1+1
    p1_1 = img[y1, x1]  # 获取 (x1, y1) 位置的像素值
    p1_2 = img[y2, x1]  # 获取 (x1, y2) 位置的像素值
    p2_1 = img[y1, x2]  # 获取 (x2, y1) 位置的像素值
    p2_2 = img[y2, x2]  # 获取 (x2, y2) 位置的像素值

    pix = (1-u)*(1-v)*p1_1 + (1-u)*v*p1_2 + u*(1-v)*p2_1 + u*v*p2_2
    return int(np.round(pix))


#def horizontal(sticker, zstore):
def horizontal(sticker,params):
    '''
    transform the picture according to parabola in horizontal direction
    input:
        sticker: Image type
        height: matrix (store height information for each coordinate)
    output:
        hor_sticker
    '''
    w, h = sticker.shape
    c, hsegment, stride, locate = params[0],params[1],params[2],params[3]
    # c = 100
    # hsegment = 150
    # stride = c
    # locate = 1
        
    a = solve_a(hsegment,stride)
    b, wn = solve_b(a,c,w,locate)
    #print('a,b,c,wn,w = ',a,b,c,wn,w)
    mask = np.zeros((h, wn), dtype=np.uint8)
    top3 = np.ones((h,wn,3))*255
    top4 = np.zeros((h,wn,1))
    newimg = np.concatenate((top3,top4),axis=2)
    newimg = Image.fromarray(np.uint8(newimg))
    
    def f(x):
        return (1 + ((x-c)*(2*a))**2)**0.5
    x_arc = [integrate.quad(f,0,xnow+1)[0]  for xnow in range(wn)]
    z = np.zeros((1,wn))

    def zfunction(x):
        return a * ((x-c)**2) + b

    for i in range(wn):
        x_map =  min(x_arc[i],w-1)
        z[0][i] = zfunction(i)
        #print(x_map,jstart,int(jstart))
        for j in range(h):
            #y_map = j+jstart-np.floor(jstart)
            #y_map = j+np.modf(jstart)[0]
            y_map = j
            #print(j,jstart,np.floor(jstart),y_map)
            #print(x_map,y_map)
            pix = bilinear_interpolation(sticker,x_map,y_map)
            if pix != 0:
                mask[j,i] = 255
            
    return mask ,z



'''
def horizontal(sticker, params):
    w, h = sticker.shape
    c, hsegment, stride, locate = params[0], params[1], params[2], params[3]
        
    a = solve_a(hsegment, stride)
    b, wn = solve_b(a, c, w, locate)
    
    # Create an empty mask to store the modified result
    mask = np.zeros((h, wn), dtype=np.uint8)
    
    def f(x):
        return (1 + ((x - c) * (2 * a)) ** 2) ** 0.5

    # Precompute the arc length for each position
    x_arc = [integrate.quad(f, 0, xnow + 1)[0] for xnow in range(wn)]
    
    def zfunction(x):
        return a * ((x - c) ** 2) + b

    # Iterate through the width of the transformed image
    for i in range(wn):
        x_map = min(x_arc[i], w - 1)
        z_val = zfunction(i)  # Calculate z-value (height information) for the current x position

        for j in range(h):
            # Use bilinear interpolation to get the pixel value from the original image
            pix = bilinear_interpolation(sticker, x_map, j)
            
            # Modify the mask based on the pixel value
            # You can set the mask value based on your logic, for example:
            if pix != 0:  # Assuming non-zero values are part of the transformed area
                mask[j, i] = 255  # Mark the corresponding position in the mask as 255 (white)
            
    return mask  # Return only the modified mask
'''

def bilinear_interpolation(img, x, y):
    w, h = img.shape
    xset = math.modf(x)
    yset = math.modf(y)
    u, v = xset[0], yset[0]
    x1, y1 = int(xset[1]), int(yset[1])
    x2 = x1 + 1 if u > 0 and x1 + 1 < w else x1
    y2 = y1 + 1 if v > 0 and y1 + 1 < h else y1
    p1_1 = img[y1, x1]  # 获取 (x1, y1) 位置的像素值
    p1_2 = img[y2, x1]  # 获取 (x1, y2) 位置的像素值
    p2_1 = img[y1, x2]  # 获取 (x2, y1) 位置的像素值
    p2_2 = img[y2, x2]  # 获取 (x2, y2) 位置的像素值

    pix = (1 - u) * (1 - v) * p1_1 + (1 - u) * v * p1_2 + u * (1 - v) * p2_1 + u * v * p2_2
    return np.round(pix).astype(np.int32)

def pitch(newimg, z, theta):
    w, h = newimg.shape
    
    # 构建俯仰变换矩阵
    m = np.array([[1, 0, 0],
                  [0, math.cos(theta), -math.sin(theta)],
                  [0, math.sin(theta), math.cos(theta)]])
    invm = np.linalg.inv(m)  # 求逆矩阵

    x = np.array(range(w))
    y1, y2 = np.ones([1, w]) * 0, np.ones([1, w]) * (h - 1)
    
    # 创建底部和顶部的坐标点
    first = np.vstack([x, y1, z]).T
    last = np.vstack([x, y2, z]).T
    
    # 进行变换
    pfirst = first.dot(m)
    plast = last.dot(m)
    
    # 计算变换后的图像高度与偏移量
    hn = int(np.floor(np.max(plast[:, 1])) - np.ceil(np.min(pfirst[:, 1]))) + 1
    shifting = np.ceil(np.min(pfirst[:, 1]))
    
    # 初始化空图像，只有单通道
    endimg = np.zeros((hn, w), dtype=np.uint8)  # 使用单通道
    
    # 计算变换后的y坐标范围
    start = np.ceil(pfirst[:, 1] - shifting)
    stop = np.floor(plast[:, 1] - shifting)

    # 对每个像素进行俯仰变换
    for i in range(w):
        jstart = int(start[i])
        jstop = int(stop[i])
        
        def zconvert(y):
            parm = binary_equation(pfirst[i][1], pfirst[i][2], plast[i][1], plast[i][2])
            return parm[0] * y + parm[1]

        for j in range(jstart, jstop + 1):
            raw_y = j + shifting
            raw_z = zconvert(raw_y)
            mapping = np.array([i, raw_y, raw_z]).dot(invm)

            # 使用双线性插值获取变换后的像素值
            pix = bilinear_interpolation(newimg, mapping[0], mapping[1])
            endimg[j, i] = pix  # 将像素值赋给新的图像

    # 返回变换后的图像和偏移量
    return endimg, shifting

def change_sticker(sticker,scale):
    new_weight = int(sticker.size[0]/scale)
    new_height = int(sticker.size[1]/scale)
    #print(new_weight,new_height)
    sticker = sticker.resize((new_weight,new_height),Image.Resampling.LANCZOS)
    return sticker


def deformation3d(sticker,operate_sticker,magnification,zstore,x,y):
    w, h = sticker.shape
    print(w)
    print(h)
    area = zstore[y-h/2:y+h/2,x-w/2:x+w/2]
    #print('y,x=',y,x,'w,h=',w,h,area.shape)
    index = np.argmax(area)
    highesty = index // area.shape[1]   # Location coordinates of the highest point in the selected area
    highestx = index % area.shape[1]
    locate = 1 if highestx > area.shape[1]/2 else 2 # =1 if the highest point is to the right
    sign = 1 if highesty < area.shape[0]/2 else -1  # =1 if the highest point is on the top(Forward rotation)
    c = highestx
    if (locate==1):
        hsegment = area[highesty][highestx] - area[highesty][0]
        stride = c
    elif(locate==2):
        #hsegment = area[highesty][highestx] - area[highesty][w-1]
        hsegment = area[highesty][highestx] - area[highesty][area.shape[1]-1]
        stride = w - highestx
    
    #step = 10
    if (sign==1):
        step = max(min(20,area.shape[0]-highesty-1),1)
        #print('area.shape =',area.shape,'y,x=',highesty,highestx,'step=',step)
        partz = area[highesty][highestx] - area[highesty+step][highestx]
        party = step
        theta = min(math.atan(partz/party),math.radians(40))
        #theta = math.atan(partz/party)
    elif(sign==-1):
        step = max(min(20,highesty),1)
        partz = area[highesty][highestx] - area[highesty-step][highestx]
        party = step
        theta = max(-1 * math.atan(partz/party),math.radians(-40))
        #theta = -1 * math.atan(partz/party)
    operate_params = [c*magnification,hsegment,stride*magnification,locate]
    # print(operate_params)
    # print('theta = ',math.degrees(theta))
    newimg,z = horizontal(operate_sticker,operate_params)
    endimg,shifting = pitch(newimg,z,theta/2)
    #endimg.show()
    # if(sign==-1):
    #     y = y + int(shifting)
    #sticker = stick.transparent_back(endimg)
    #print(sticker.size)
    #sticker.show()
    sticker=change_sticker(endimg,magnification)

    return sticker,y
