import os
import time
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import DataLoader
from maketable import *
from makedata import * 
from model import *

np.set_printoptions(threshold=0.1)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

SUBDIVISION = 3
NUM_CLASS = 10

class SpherePHD():

    def autoencoder(self, subdivision):
        self.conv_tables = []
        self.adj_tables = []
        self.upsample_tables = []
        self.pooling_tables = make_pooling_table(subdivision+1)
        for div in range(0, subdivision+1):
            conv_table = make_conv_table(div)
            self.conv_tables.append(conv_table)
            adj_table = make_adjacency_table(div)
            self.adj_tables.append(adj_table)
            if div > 0:
                upsample_table = make_upsample_table(self.pooling_tables[div-1], adj_table)
                self.upsample_tables.append(upsample_table)
        self.image_size = 20 * 4 ** subdivision 

        self.num_epochs = 500
        self.num_steps = 150
        self.batch_size = 1
        self.X = tf.placeholder(tf.float32, [None, 1, 20 * 4**subdivision, 3])
        self.Y = tf.placeholder(tf.float32, [None, 20 * 4**subdivision, NUM_CLASS])

        self.logits = auto_encoder(self.X, self.conv_tables, self.adj_tables,\
                                   self.pooling_tables, self.upsample_tables, subdivision)
        self.prediction = tf.nn.softmax(self.logits)
        self.prediction = tf.squeeze(self.prediction, axis=1)

        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.train_op = self.optimizer.minimize(self.loss_op)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(self.prediction, 2), tf.argmax(self.Y, 2))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.saver = tf.train.Saver()

        # Start training
        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = 1
        self.config.gpu_options.allow_growth = True

    def simple_cnn(self, subdivision):
        self.conv_tables = []
        self.adj_tables = []
        for i in range(0, subdivision+1):
            self.conv_tables.append(make_conv_table(i))
            self.adj_tables.append(make_adjacency_table(i))
            
        self.pooling_tables = make_pooling_table(subdivision+1)
        self.image_size = 20 * 4 ** subdivision 
        self.subdivision = subdivision
        self.num_epochs = 10
        self.num_steps = 3000
        self.batch_size = 20
        self.model = MNIST_net(self.conv_tables, self.adj_tables, self.pooling_tables, subdivision)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-1)  # lr hand-tuned
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(self.model.parameters)
        
    def train(self, restore=False):
        batch_size = self.batch_size
        device = self.device
        optimizer = self.optimizer
        loss_function = self.loss_function
        epochs = self.num_epochs
        model = self.model
        
        train_data = np.array(list(np.transpose(np.load("./train_data.npy"),(0,2,1))[:,:,None,:]))
        train_label = np.array(list(np.load("./train_label.npy")))
        test_data = np.array(list(np.transpose(np.load("./test_data.npy"),(0,2,1))[:,:,None,:]))
        test_label = np.array(list(np.load("./test_label.npy")))   
        
        print("image shape: ",train_data.shape)
        train_dataset = TensorDataset(torch.Tensor(train_data),torch.tensor(train_label,dtype=torch.long))
        test_dataset = TensorDataset(torch.Tensor(test_data),torch.tensor(test_label,dtype=torch.long))
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        ##Train
        start = time.time()
        trainloss = []
        
        for epoch in range(self.num_epochs) :
            loss_sum = 0
            print("{}th epoch starting.".format(epoch))
            with tqdm(total=train_data.shape[0]/self.batch_size) as pbar:
                for i, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(device), labels.to(device)

                    optimizer.zero_grad()
                    output = model(images)

                    train_loss = loss_function(output, labels)
                    loss_sum += train_loss.item()
                    train_loss.backward()

                    self.optimizer.step()
                    pbar.update(1)
                
                trainloss.append(loss_sum/train_data.shape[0])

            print("Training Loss: {}".format(trainloss[-1]))
                
                
        end = time.time()
        print("Time ellapsed in training is: {}".format(end - start))
        
        ##Test
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        
        
        test_loss, correct, total = 0, 0, 0
        for images, labels in test_loader :
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            test_loss += loss_function(output, labels).item()

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

            total += labels.size(0)
            
        print('[Test set] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss /total, correct, total, 100. * correct / total))
        print('Model saved in path: %s' % save_path)
        print('Training finished')

    def print_test(self, sess, epoch, save_img=False):

        result = []
        batch_size = self.batch_size
        total_accuracy = 0
        test_images = np.load('./area_1.npy')[150:]
        test_labels = np.load('./area_1_label.npy')[150:]
        test_one_hot = np.zeros((test_labels.shape[0], test_labels.shape[1], 14))
        for i in range(test_labels.shape[0]):
            for j in range(test_labels.shape[1]):
                test_one_hot[i, j, int(test_labels[i, j])] = 1
        
        for step in range(1, 41):
            batch_x, batch_y = np.reshape(test_images[(step - 1) * batch_size:step * batch_size].astype(np.float32),
                                          [batch_size, 1, self.image_size, 3]),\
                                          test_one_hot[(step - 1) * batch_size:step * batch_size].astype(np.float32)
            acc = sess.run([self.accuracy, self.prediction], feed_dict={self.X: batch_x, self.Y: batch_y})
            total_accuracy = total_accuracy + acc[0]
            result.append(np.array(acc[1]))
        total_accuracy *= batch_size
        print('Testing accuracy is {:.4f}'.format(total_accuracy/40))

        result = np.array(result)
        np.save('prediction.npy', result)

        if save_img:
            img = result[0].reshape(-1, 14)
            icosa2pano(img, 'sub_6.log', './reconstruct_image/segment_'+str(epoch)+'.jpg', paint_seg=True)

    def test(self):

        result = []
        batch_size = self.batch_size
        sess = tf.Session(config=self.config)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        self.saver.restore(sess, './save/Stanford2D3D_Sphere_model_149.ckpt')
        #print("Testing start")
        total_accuracy = 0
        for idx in range(1,2):
            test_images = np.load('./area_1.npy')[150:]
            test_labels = np.load('./area_1_label.npy')[150:]
            for step in range(1, 41):
                batch_x, batch_y = np.reshape(test_images[(step - 1) * batch_size:step * batch_size].astype(np.float32),
                                              [batch_size, 1, 1280, 1]), test_labels[
                                              (step - 1) * batch_size:step * batch_size].astype(np.float32)
                acc = sess.run([self.accuracy, self.prediction], feed_dict={self.X: batch_x, self.Y: batch_y})
                total_accuracy = total_accuracy + acc[0]
                result.apppend(np.array(acc[1]))
        total_accuracy *= batch_size
        print('Total accuracy is {:.4f}'.format(total_accuracy/40))
        sess.close()

        result = np.array(result)
        np.save('result_ver1.npy', result)

def main():
    
    MNIST = SpherePHD()
    MNIST.simple_cnn(SUBDIVISION)
    MNIST.train()
    MNIST.test()
    '''
    Stanford2D3D = SpherePHD()
    Stanford2D3D.autoencoder(SUBDIVISION)
    Stanford2D3D.train()
	'''
if __name__ == '__main__':
    main()
