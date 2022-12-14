import os
import numpy as np
import cv2
import pickle
from tqdm import tqdm
#from utils import *
from makedata import *
from maketable import *

CLASS_LIST = ['<UNK>', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter',
              'column', 'door', 'floor', 'sofa', 'table', 'wall', 'window']
ID_LIST = [1, 160, 297, 881, 1266, 2629, 6511, 6765, 7308, 7592, 7647, 8102, 9648, 9816]
np.random.seed(1513)

class DataLoader():
    
    def __init__ (self, dir_path, subdivision):
        # Process the panoramic picture
        self.dataset = []
        self.label = []
        cnt = 1
        dict = unpickle(dir_path)
        
        self.label = dict[b'fine_labels']
        imgs = dict[b'data']
        
        with tqdm(total=imgs.shape[0]) as pbar:
            for row_img in imgs:
                img = row_img.reshape(3,32,32).transpose(1,2,0)
                self.dataset.append(pano2icosa(img, subdivision))
                pbar.update(1)
        '''
        for num in range(10):
            for file in sorted(os.listdir(dir_path+"/"+str(num)+"/")):
                if file[-4:] == '.png' or file[-4:] == '.jpg':
                    # subdivision defualt = 8
                    img = cv2.imread(os.path.join(dir_path+"/"+str(num)+"/", file))
                    #img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)  ## FOR MNIST
                    img = cv2.resize(img, (100, 100))
                    self.dataset.append(pano2icosa(img, subdivision))
                    print('Process %05d pictures..\r' %cnt, end='')
                    cnt += 1
                    self.label.append(num)
'''
        self.dataset = np.array(self.dataset) # shape = (N, 4**8 * 20, 3)
        self.dataset = self.dataset / 255
        self.label = np.array(self.label)
        self.num_data = self.dataset.shape[0]
'''
    def shuffle(self):
        self.rng = np.random.shuffle(np.arange(self.num_data))
        self.dataset = self.dataset[self.rng]

    def make_label(self, subdivision):
        reference = load_labels('./assets/semantic_labels.json')
        cnt = 1
        ret = []
        for file in sorted(os.listdir('/media/bl530/???????????????/area_1/pano/semantic/')):
            if file[-4:] == '.png' or file[-4:] == '.jpg':
                img = cv2.imread(os.path.join('/media/bl530/???????????????/area_1/pano/semantic/', file))
                img = cv2.resize(img, (400, 200))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tmp = np.zeros((img.shape[0], img.shape[1], 1))-1
                print('Label %05d pictures..\r' %cnt, end='')
                index = img[:, :, 0]*256*256 + img[:, :, 1]*256 + img[:, :, 2]
                index = np.expand_dims(index, axis=2)
                index[index > 9816] = 0
                for i in range(len(CLASS_LIST)):
                    tmp[(index < ID_LIST[i]) & (tmp == -1)] = i
                ret.append(pano2icosa(tmp, subdivision))
                cnt += 1
        print('')
        return np.array(ret)
'''
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def main():
    train_data = DataLoader('./dataset/cifar-100-python/train', 4)
    np.save('./cifar100_train_data.npy', train_data.dataset)
    np.save('./cifar100_train_label.npy',train_data.label)
    print('')
    del train_data
    test_data = DataLoader('./dataset/cifar-100-python/test', 4)
    np.save('./cifar100_test_data.npy', test_data.dataset)
    np.save('./cifar100_test_label.npy', test_data.label)
    print('')

if __name__ == '__main__':
    main()
