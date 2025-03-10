from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt

def transparent_back(pic_path):    # make jpg picture's background transparent(png)
    img = cv2.imread(pic_path)     # array
    sticker = Image.open(pic_path) # image
    W,H = sticker.size
    #print(W,H)
    mask = np.zeros(img.shape[:2],np.uint8)
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (1,1,450,450)
    
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT) 
    #print(mask)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8').transpose()
    print('mask2 = ',mask2.shape)

    #print(mask2[200][200])
    sticker = sticker.convert('RGBA')
    for i in range(W):
        for j in range(H):
            color_1 = sticker.getpixel((i,j))
            if(mask2[i][j]==0):   # transparent
                color_1 = color_1[:-1] + (0,)
                sticker.putpixel((i,j),color_1)
            else:
                color_1 = color_1[:-1] + (255,)
                sticker.putpixel((i,j),color_1)
    sticker.show()
    sticker.save(pic_path[:-3]+'png')

def make_stick2(args, backimg,sticker,x,y,factor=1):
    '''
    backimg = np.array(backimg)
    backimg = cv2.cvtColor(backimg,cv2.COLOR_GRAY2RGB)
    r,g,b = cv2.split(backimg)
    background = cv2.merge([b, g, r])
    #print('background = ',background.shape)
    base,_ = make_basemap(background.shape[1],background.shape[0],sticker,x=x,y=y)
    #print('basemap = ',basemap.shape)
    #print('basemap = ',basemap[100][130][3])
    r,g,b,a = cv2.split(base)
    foreGroundImage = cv2.merge([b, g, r,a])
    # cv2.imshow("outImg",foreGroundImage)
    # cv2.waitKey(0)

    b,g,r,a = cv2.split(foreGroundImage)
    foreground = cv2.merge((b,g,r))
    
    alpha = cv2.merge((a,a,a))

    foreground = foreground.astype(float)
    background = background.astype(float)
    
    alpha = alpha.astype(float)/255
    alpha = alpha * factor
    #print('alpha = ',alpha)
    
    foreground = cv2.multiply(alpha,foreground)
    background = cv2.multiply(1-alpha,background)
    
    outarray = foreground + background
    #cv2.imwrite("outImage.jpg",outImage)

    # cv2.imshow("outImg",outImage/255)
    # cv2.waitKey(0)
    b, g, r = cv2.split(outarray)
    outarray = cv2.merge([r, g, b])
    outarray = np.clip(outarray,0,255).astype(np.uint8)
    outImage_gray = cv2.cvtColor(outarray, cv2.COLOR_RGB2GRAY)
    outImage = Image.fromarray(np.uint8(outImage_gray))
    '''
    backimg = np.array(backimg)
    backimg = cv2.cvtColor(backimg, cv2.COLOR_GRAY2RGB)
    r, g, b = cv2.split(backimg)
    background = cv2.merge([b, g, r])
    base, basemap = make_basemap(background.shape[1], background.shape[0], sticker, x=x, y=y)
    r, g, b, a = cv2.split(base)
    foreGroundImage = cv2.merge([b, g, r, a])

    b, g, r, a = cv2.split(foreGroundImage)
    foreground = cv2.merge((b, g, r))
    alpha = cv2.merge((a, a, a)) 

    foreground = foreground.astype(float)
    background = background.astype(float)

    alpha = alpha.astype(float)/255
    alpha = alpha * factor
    if args.data_type == 'casia' or args.data_type == 'buua':
        foreground[basemap==1] = background[basemap==1] * 0.416
    else:
        foreground[basemap==1] = background[basemap==1] * 0.33
    
    foreground = cv2.multiply(alpha,foreground)
    background = cv2.multiply(1-alpha,background)
    outarray = foreground + background

    b, g, r = cv2.split(outarray)
    outarray = cv2.merge([r, g, b])
    outarray = np.clip(outarray,0,255).astype(np.uint8)
    outImage_gray = cv2.cvtColor(outarray, cv2.COLOR_RGB2GRAY)
    outImage = Image.fromarray(np.uint8(outImage_gray))

    return outImage

def change_sticker(sticker,scale=1):
    #sticker = Image.open(stickerpath)
    new_weight = int(sticker.size[0]/scale)
    new_height = int(sticker.size[1]/scale)
    #print(new_weight,new_height)
    sticker = sticker.resize((new_weight,new_height),Image.LANCZOS)
    return sticker

def make_basemap(width,height,sticker,x,y):
    layer = Image.new('RGBA',(width,height),(255,255,255,0)) # white and transparent
    layer.paste(sticker,(x,y))
    #layer.show()
    base = np.array(layer)
    alpha_matrix = base[:,:,3]
    basemap = np.where(alpha_matrix!=0,1,0)
    return base,basemap

if __name__ == '__main__':
   #def make_stick2(backimg,sticker,x,y,factor=1): 
   backimg = Image.open('/home/qiuchengyu/mynewproject/adv-nir_face/s1_VIS_00001_004.jpg')
   sticker = Image.open('/home/qiuchengyu/mynewproject/adv-nir_face/black_image.png')
   x = 50
   y = 90
   factor = 1
   scale = 40
   sticker = change_sticker(sticker, scale)
   outimg = make_stick2(backimg, sticker, x, y, factor)
   outimg.save('outimg.png')