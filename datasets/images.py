import cv2, math
import numpy as np

def jitter(image, ratio=0.2):
    '''
    equal to random crop
    :image          opencv mat object, or numpy object
    :ratio          maximum ratio of edge to cut off 
    :return         opencv mat
    '''
    w, h, _ = image.shape
    cut_w = int(w*ratio)
    cut_h = int(h*ratio)
    
    start_w = np.random.randint(0, cut_w)
    start_h = np.random.randint(0, cut_h)
    piece_image = image[start_h:h-cut_h, start_w:w-cut_w, :]
    return piece_image


def intensity(image):
    '''
    add intensity change by modify HSV channel
    :image          opencv mat object, or numpy object
    :return         openc mat
    '''
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV);
    hsv[:,:,0] = hsv[:,:,0]*(0.8+ np.random.random()*0.2);
    hsv[:,:,1] = hsv[:,:,1]*(0.3+ np.random.random()*0.7);
    hsv[:,:,2] = hsv[:,:,2]*(0.2+ np.random.random()*0.8);
    img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR);
    return img


def flip(image):
    '''
    flip image from left to right
    :image          opencv mat object, or numpy object
    :return         openc mat
    '''
    return np.fliplr(image)


def blur(image, maxlevel=3):
    '''
    Add Gaussian Blur
    :image          opencv mat object, or numpy object
    :level          int, to control the blur status
    :return         openc mat
    '''
    level = np.random.randint(1,maxlevel+1)
    return cv2.blur(image, (level * 2 + 1, level * 2 + 1));

def rotate(image, angle_bound=5, padding_value=(0,0,0)):
    '''
    rotate images, padding value can be set
    :image          opencv mat object, or numpy object
    :angle          int, from 0 to 360
    :padding_value  RGB tuple
    :return         openc mat    
    '''
    angle = np.random.randint(-angle_bound, angle_bound+1)
    w = image.shape[1]
    h = image.shape[0]
    scale = 1
    radium = np.deg2rad(angle)
    new_w = (abs(np.cos(radium)) * w + abs(np.sin(radium)) * h) 
    new_h = (abs(np.cos(radium)) * h + abs(np.sin(radium)) * w)
    rot_mat = cv2.getRotationMatrix2D((new_w*0.5, new_h*0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(new_w-w)*0.5, (new_h-h)*0.5,0]))
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(image, rot_mat, (int(math.ceil(new_w)), int(math.ceil(new_h))), flags=cv2.INTER_LANCZOS4, borderValue=padding_value)


def noise(image):
    '''
    Add Guassian Noise
    :image          opencv mat object, or numpy object
    :return         openc mat    
    '''
    def AddNoiseSingleChannel(single):
        diff = 255-single.max();
        noise = np.random.normal(0,1+r(6),single.shape);
        noise = (noise - noise.min())/(noise.max()-noise.min())
        noise= diff*noise;
        noise= noise.astype(np.uint8)
        dst = single + noise
        return dst
    img = image.copy()
    img[:,:,0] =  AddNoiseSingleChannel(img[:,:,0]);
    img[:,:,1] =  AddNoiseSingleChannel(img[:,:,1]);
    img[:,:,2] =  AddNoiseSingleChannel(img[:,:,2]);
    return img



def color_preprocessing(x_train):
    '''
    normalize images 
    '''
    x_train = x_train.astype('float32')
    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])
    return x_train















































