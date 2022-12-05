from layer import *
import torch.nn as nn

class MNIST_net(nn.Module) :
    def __init__(self, conv_tables, adj_tables, pooling_tables, div) :
        super(MNIST_net, self).__init__()
        self.conv_tables = conv_tables
        self.adj_tables = adj_tables
        self.pooling_tables = pooling_tables
        self.div = div
        self.conv1 = PHD_conv2d(in_dim=3, out_dim=16, stride=1)
        #self.pool1 = PHD_maxpool()
        self.conv2 = PHD_conv2d(in_dim=16, out_dim=32, stride=1)
        #self.pool2 = PHD_maxpool()
        self.conv3 = PHD_conv2d(in_dim=32, out_dim=10, stride=1)
        
    def forward(self, x) :
        div = self.div
        x = self.conv1(x, conv_table=self.conv_tables[div].T)
        x = PHD_maxpool(x, self.adj_tables[div].T, self.pooling_tables[div-1].T)
        x = self.conv2(x, conv_table=self.conv_tables[div-1].T)
        x = PHD_maxpool(x, self.adj_tables[div-1].T, self.pooling_tables[div-2].T)
        x = self.conv3(x, conv_table=self.conv_tables[div-2].T)
        out = PHD_avgpool(x)
        return out

class CIFAR_net(nn.Module) :
    def __init__(self, conv_tables, adj_tables, pooling_tables, div) :
        super(CIFAR_net, self).__init__()
        self.conv_tables = conv_tables
        self.adj_tables = adj_tables
        self.pooling_tables = pooling_tables
        self.div = div
        self.conv1 = PHD_conv2d(in_dim=3, out_dim=32, stride=1)
        #self.pool1 = PHD_maxpool()
        self.conv2 = PHD_conv2d(in_dim=32, out_dim=128, stride=1)
        #self.pool2 = PHD_maxpool()
        self.conv3 = PHD_conv2d(in_dim=128, out_dim=100, stride=1)
        
    def forward(self, x) :
        div = self.div
        x = self.conv1(x, conv_table=self.conv_tables[div].T)
        x = PHD_maxpool(x, self.adj_tables[div].T, self.pooling_tables[div-1].T)
        x = self.conv2(x, conv_table=self.conv_tables[div-1].T)
        x = PHD_maxpool(x, self.adj_tables[div-1].T, self.pooling_tables[div-2].T)
        x = self.conv3(x, conv_table=self.conv_tables[div-2].T)
        out = PHD_avgpool(x)
        return out



'''
def MNIST_net(x, conv_tables, adj_tables, pooling_tables, div):
    conv1 = PHD_conv2d(x, in_dim=1, out_dim=16, conv_table=conv_tables[div].T, name='conv_1')
    pool1 = PHD_maxpool(conv1, adj_tables[div].T, pooling_tables[div-1].T)

    conv2 = PHD_conv2d(pool1, in_dim=16, out_dim=32, conv_table=conv_tables[div-1].T, name='conv_2')
    pool2 = PHD_maxpool(conv2, adj_tables[div-1].T, pooling_tables[div-2].T)

    conv3 = PHD_conv2d(pool2, in_dim=32, out_dim=10, conv_table=conv_tables[div-2].T, name='conv_3')
    out = PHD_avgpool(conv3)
    return out


def MNIST_net(x, conv_tables, adj_tables, pooling_tables, div):

    conv1 = conv2d(x, in_dim=1, out_dim=16, conv_table=conv_tables[div].T, name='conv_1')
    pool1 = maxpool(conv1, adj_tables[div].T, pooling_tables[div-1].T)

    conv2 = conv2d(pool1, in_dim=16, out_dim=32, conv_table=conv_tables[div-1].T, name='conv_2')
    pool2 = maxpool(conv2, adj_tables[div-1].T, pooling_tables[div-2].T)

    conv3 = conv2d(pool2, in_dim=32, out_dim=10, conv_table=conv_tables[div-2].T, name='conv_3')
    out = avgpool(conv3)

    return out



def auto_encoder(x, conv_tables, adj_tables, pooling_tables, upsample_tables, div):

    conv1 = conv2d(x, in_dim=3, out_dim=128, conv_table=conv_tables[div].T, name='block1_conv1')
    pool1 = maxpool(conv1, adj_tables[div].T, pooling_tables[div-1].T)
    conv2 = conv2d(pool1, in_dim=128, out_dim=256, conv_table=conv_tables[div-1].T, name='block2_conv1')
    pool2 = maxpool(conv2, adj_tables[div-1].T, pooling_tables[div-2].T)
    conv3 = conv2d(pool2, in_dim=256, out_dim=512, conv_table=conv_tables[div-2].T, name='block3_conv1')

    latent = channel_conv(conv3, in_dim=512, out_dim=256, name='channel_conv')
    upsample1 = upsample(latent, upsample_tables[div-2].T)
    de_conv2 = conv2d(upsample1, in_dim=256, out_dim=128, conv_table=conv_tables[div-1].T, name='block4_deconv1')
    upsample2 = upsample(de_conv2, upsample_tables[div-1].T)
    de_conv1 = conv2d(upsample2, in_dim=128, out_dim=14, conv_table=conv_tables[div].T, name='block5_deconv1')
    return de_conv1
'''
