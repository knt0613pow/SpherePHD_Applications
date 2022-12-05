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



class PHD_VGG16(nn.Module):
    def __init__(self, conv_tables, adj_tables, pooling_tables, div):
        super(PHD_VGG16, self).__init__()
        self.conv_tables = conv_tables
        self.adj_tables = adj_tables
        self.pooling_tables = pooling_tables
        self.div = div

        self.layer1 = nn.Sequential(
            PHD_conv2d(in_dim = 3, out_dim =64, stride = 1, conv_table = conv_tables[div]),
            PHD_conv2d(in_dim = 64, out_dim = 64, stride = 1, conv_table = conv_tables[div]),
        )
        self.layer2 = nn.Sequential(
            PHD_conv2d(in_dim = 64, out_dim =128, stride = 1, conv_table = conv_tables[div-1]),
            PHD_conv2d(in_dim = 128, out_dim = 128, stride = 1, conv_table = conv_tables[div-1]),
        )
        self.layer3 = nn.Sequential(
            PHD_conv2d(in_dim = 128, out_dim =256, stride = 1, conv_table = conv_tables[div-2]),
            PHD_conv2d(in_dim = 256, out_dim = 256, stride = 1, conv_table = conv_tables[div-2]),
            PHD_conv2d(in_dim = 256, out_dim = 256, stride = 1, conv_table = conv_tables[div-2]),
        )

        self.layer4 = nn.Sequential(
            PHD_conv2d(in_dim = 256, out_dim =512, stride = 1, conv_table = conv_tables[div-3]),
            PHD_conv2d(in_dim = 512, out_dim = 512, stride = 1, conv_table = conv_tables[div-3]),
            PHD_conv2d(in_dim = 512, out_dim = 512, stride = 1, conv_table = conv_tables[div-3]),
        )

        self.layer5 = nn.Sequential(
            PHD_conv2d(in_dim = 512, out_dim =512, stride = 1, conv_table = conv_tables[div-4]),
            PHD_conv2d(in_dim = 512, out_dim = 512, stride = 1, conv_table = conv_tables[div-4]),
            PHD_conv2d(in_dim = 512, out_dim = 512, stride = 1, conv_table = conv_tbales[div-4]),
        )
    def forward(self, x):
        l1 = self.layer1(x)
        #adj_table, pooling_table
        l1 = PHD_maxpool(l1, adj_table[div], pooling_table[div])
        l2 = self.layer2(l1)
        l2 = PHD_maxpool(l2, adj_table[div-1], pooling_table[div-1])
        l3= self.layer3(l2)
        l3 = PHD_maxpool(l3, adj_table[div-2], pooling_table[div-2])
        l4= self.layer3(l3)
        l4 = PHD_maxpool(l4, adj_table[div-3], pooling_table[div-3])
        l5= self.layer3(l4)
        l5 = PHD_maxpool(l5, adj_table[div-4], pooling_table[div-4])
        return l5



if __name__ == '__main__':
    subdivision = 5
    conv_tables = []
    adj_tables = []
    for i in range(0, subdivision+1):
        conv_tables.append(make_conv_table(i))
        adj_tables.append(make_adjacency_table(i))
        
    pooling_tables = make_pooling_table(subdivision+1)
    model = PHD_VGG16(conv_tables, adj_tables, pooling_tables, subdivision)