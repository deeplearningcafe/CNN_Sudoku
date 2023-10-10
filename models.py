import torch.nn as nn

class CNNModel_BaseModelV3(nn.Module):
    def __init__(self, n_channels=32):
        super(CNNModel_BaseModelV3, self).__init__()
        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_channels, kernel_size=3, stride=1, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_channels)
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_channels * 2)
        self.conv3 = nn.Conv2d(in_channels=n_channels * 2, out_channels=n_channels * 4, kernel_size=3, stride=1, padding=1)
        self.conv3_batchnorm = nn.BatchNorm2d(num_features=n_channels * 4)
        self.conv4 = nn.Conv2d(in_channels=n_channels * 4, out_channels=n_channels * 8, kernel_size=3, stride=1, padding=1)
        self.conv4_batchnorm = nn.BatchNorm2d(num_features=n_channels * 8)

        self.activation = nn.Tanh()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1 * 1 * self.n_channels * 8, 128)
        self.fc2 = nn.Linear(128, 81)

    def forward(self, x):
        print("shape x:", x.shape)
        x = self.conv1_batchnorm(self.conv1(x))
        x = self.pool(self.activation(x))
        x = self.conv2_batchnorm(self.conv2(x))
        x = self.pool(self.activation(x))
        print("shape of x: ", x.shape)
        x = self.conv3_batchnorm(self.conv3(x))
        x = self.activation(x)
        x = self.conv4_batchnorm(self.conv4(x))
        x = self.pool(self.activation(x))
        print("shape of x: ", x.shape)

        x = x.view(-1, 1 * 1 * self.n_channels * 8)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x
    
    
    

class CNNModel_Base(nn.Module):
    def __init__(self, n_channels=32):
        super(CNNModel_Base, self).__init__()
        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_channels, kernel_size=3, stride=1, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_channels)
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_channels * 2)
        self.conv3 = nn.Conv2d(in_channels=n_channels * 2, out_channels=n_channels * 4, kernel_size=3, stride=1, padding=1)
        self.conv3_batchnorm = nn.BatchNorm2d(num_features=n_channels * 4)
        self.conv4 = nn.Conv2d(in_channels=n_channels * 4, out_channels=n_channels * 8, kernel_size=3, stride=1, padding=1)
        self.conv4_batchnorm = nn.BatchNorm2d(num_features=n_channels * 8)
        self.last_conv = nn.Conv2d(in_channels=n_channels * 8, out_channels=9, kernel_size=1)

        self.activation = nn.ReLU()
        self.softmax = nn.Softmax2d()


    def forward(self, x):
        #print("shape x:", x.shape)
        x = self.conv1_batchnorm(self.conv1(x))
        x = self.activation(x)
        x = self.conv2_batchnorm(self.conv2(x))
        x = self.activation(x)
        x = self.conv3_batchnorm(self.conv3(x))
        x = self.activation(x)
        x = self.conv4_batchnorm(self.conv4(x))
        x = self.activation(x)
        #print("shape before last", x.shape) ([100, 256, 9, 9])
        x = self.last_conv(x)
        #print("shape after: ", x.shape) ([100, 9, 9, 9])
        
        return x
    
    
class SudokuBlock(nn.Module):
    def __init__(self, in_channels=32, out_channels=64, kernel_size=3, bias=True):
        super(SudokuBlock, self).__init__()
        self.kernel_size = 3
        self.bias = True
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size, stride=1,
                      padding=1, bias=self.bias),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.block(x)
        #print(x.shape)
        return x
    
class CNNModel_BaseV2(nn.Module):
    def __init__(self, start_channels=32, num_blocks=2):
        super(CNNModel_BaseV2, self).__init__()
        channel_coeff = 2
        self.convs1 = nn.Sequential(
            SudokuBlock(in_channels=1, out_channels=32, kernel_size=3),
            SudokuBlock(32, 64, 3),
            SudokuBlock(64, 128, 3),
            SudokuBlock(128, 256, 3),
        )
        """SudokuBlock(in_channels=1, out_channels=start_channels*(channel_coeff^1), kernel_size=3),
            SudokuBlock(start_channels*(channel_coeff^1), start_channels*(channel_coeff^2), 3),
            SudokuBlock(start_channels*(channel_coeff^2), start_channels*(channel_coeff^3), 3),
            SudokuBlock(start_channels*(channel_coeff^3), start_channels*(channel_coeff^4), 3),"""
        
        self.convs2 = nn.Sequential(
            *(num_blocks * [SudokuBlock(256, 256, 3)]))
        
        #self.last_conv = nn.Conv2d(in_channels=512, out_channels=9, kernel_size=1, stride=1, padding=1, bias=True)
        # as it is kernel 1 then if using padding, it adds 3x3
        self.last_conv = nn.Conv2d(in_channels=256, out_channels=9, kernel_size=1)

    def forward(self, x):
        x = self.convs1(x)
        x = self.convs2(x)
        x = self.last_conv(x)
        #print(x.shape)
        return x
        # the model is too big, it takes a lot of time

class CNNModel_BaseV3(nn.Module):
    def __init__(self, start_channels=32, num_blocks=4):
        super(CNNModel_BaseV3, self).__init__()
        
        self.convs1 = nn.Sequential(
            SudokuBlock(in_channels=1, out_channels=16, kernel_size=3),
            SudokuBlock(16, 32, 3),
            SudokuBlock(32, 64, 3),
            SudokuBlock(64, 128, 3),
        )
        
        self.convs2 = nn.Sequential(
            *(num_blocks * [SudokuBlock(128, 128, 3)]))
        
        #self.last_conv = nn.Conv2d(in_channels=512, out_channels=9, kernel_size=1, stride=1, padding=1, bias=True)
        # as it is kernel 1 then if using padding, it adds 3x3
        self.last_conv = nn.Conv2d(in_channels=128, out_channels=9, kernel_size=1)

    def forward(self, x):
        x = self.convs1(x)
        x = self.convs2(x)
        x = self.last_conv(x)
        return x

class CNNModel_BaseV4(nn.Module):
    def __init__(self, num_blocks=3):
        super(CNNModel_BaseV4, self).__init__()
        
        self.convs1 = nn.Sequential(
            SudokuBlock(in_channels=1, out_channels=64, kernel_size=3),
            SudokuBlock(64, 256, 3),
            SudokuBlock(256, 512, 3),
            SudokuBlock(512, 512, 3),# 1
            SudokuBlock(512, 512, 3),# 2
            SudokuBlock(512, 512, 3),# 3
        )
        
        """ self.convs2 = nn.Sequential(
            *(num_blocks * [SudokuBlock(512, 512, 3)]))"""
        
        #self.last_conv = nn.Conv2d(in_channels=512, out_channels=9, kernel_size=1, stride=1, padding=1, bias=True)
        # as it is kernel 1 then if using padding, it adds 3x3
        self.last_conv = nn.Conv2d(in_channels=512, out_channels=9, kernel_size=1)

    def forward(self, x):
        x = self.convs1(x)
        #x = self.convs2(x)
        x = self.last_conv(x)
        return x


class SudokuBlock_TH(nn.Module):
    def __init__(self, in_channels=32, out_channels=64, kernel_size=3, bias=True):
        super(SudokuBlock_TH, self).__init__()
        self.kernel_size = 3
        self.bias = True
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size, stride=1,
                      padding=1, bias=self.bias),
            nn.BatchNorm2d(num_features=out_channels),
        )
        self.activation = nn.Tanh()
    
    def forward(self, x):
        x = self.block(x)
        return self.activation(x)
    
class CNNModel_BaseV5(nn.Module):
    def __init__(self):
        super(CNNModel_BaseV5, self).__init__()
        
        self.convs = nn.Sequential(
            SudokuBlock_TH(in_channels=1, out_channels=64, kernel_size=3),
            SudokuBlock_TH(64, 128, 3),
            SudokuBlock_TH(128, 256, 3),
            SudokuBlock_TH(256, 256, 3),# 1
            SudokuBlock_TH(256, 256, 3),# 2
            SudokuBlock_TH(256, 256, 3),# 3
            SudokuBlock_TH(256, 256, 3),# 4
            SudokuBlock_TH(256, 512, 3),# 5

        )
        
        #self.last_conv = nn.Conv2d(in_channels=512, out_channels=9, kernel_size=1, stride=1, padding=1, bias=True)
        # as it is kernel 1 then if using padding, it adds 3x3
        self.last_conv = nn.Conv2d(in_channels=512, out_channels=9, kernel_size=1)

    def forward(self, x):
        x = self.convs(x)
        x = self.last_conv(x)
        return x
    
    
class CNNModel_BaseV6(nn.Module):
    def __init__(self):
        super(CNNModel_BaseV6, self).__init__()
        
        self.convs = nn.Sequential(
            SudokuBlock(in_channels=1, out_channels=64, kernel_size=3),
            SudokuBlock(64, 256, 3),
            SudokuBlock(256, 512, 3),
            SudokuBlock(512, 512, 3),# 1
            SudokuBlock(512, 512, 3),# 2
            SudokuBlock(512, 512, 3),# 3
            SudokuBlock(512, 512, 3),# 4
            SudokuBlock(512, 512, 3),# 5
        )
        
        #self.last_conv = nn.Conv2d(in_channels=512, out_channels=9, kernel_size=1, stride=1, padding=1, bias=True)
        # as it is kernel 1 then if using padding, it adds 3x3
        self.last_conv = nn.Conv2d(in_channels=512, out_channels=9, kernel_size=1)

    def forward(self, x):
        x = self.convs(x)
        x = self.last_conv(x)
        return x
    
class CNNModel_BaseV7(nn.Module):
    def __init__(self):
        super(CNNModel_BaseV7, self).__init__()
        
        self.convs = nn.Sequential(
            SudokuBlock(in_channels=1, out_channels=64, kernel_size=3),
            SudokuBlock(64, 256, 3),
            SudokuBlock(256, 512, 3),
            SudokuBlock(512, 512, 3),# 1
            SudokuBlock(512, 512, 3),# 2
            SudokuBlock(512, 1024, 3),# 3
            SudokuBlock(1024, 1024, 3),# 4
        )
        
        #self.last_conv = nn.Conv2d(in_channels=512, out_channels=9, kernel_size=1, stride=1, padding=1, bias=True)
        # as it is kernel 1 then if using padding, it adds 3x3
        self.last_conv = nn.Conv2d(in_channels=1024, out_channels=9, kernel_size=1)

    def forward(self, x):
        x = self.convs(x)
        x = self.last_conv(x)
        return x
    
class CNNModel_BaseV8(nn.Module):
    def __init__(self):
        super(CNNModel_BaseV8, self).__init__()
        
        self.convs = nn.Sequential(
            SudokuBlock(in_channels=1, out_channels=64, kernel_size=3),# 1
            SudokuBlock(64, 256, 3),# 2
            SudokuBlock(256, 512, 3),# 3
            SudokuBlock(512, 512, 3),# 4
            SudokuBlock(512, 512, 3),# 5
            SudokuBlock(512, 512, 3),# 6
            SudokuBlock(512, 512, 3),# 7
            SudokuBlock(512, 512, 3),# 8
            SudokuBlock(512, 512, 3),# 9
            SudokuBlock(512, 512, 3),# 10

        )
        
        #self.last_conv = nn.Conv2d(in_channels=512, out_channels=9, kernel_size=1, stride=1, padding=1, bias=True)
        # as it is kernel 1 then if using padding, it adds 3x3
        self.last_conv = nn.Conv2d(in_channels=512, out_channels=9, kernel_size=1)# 11

    def forward(self, x):
        x = self.convs(x)
        x = self.last_conv(x)
        return x