import torch
import torch.nn as nn
import math

#op = OPS[primitive](C, stride, False)

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool1d(3, stride=stride, padding=1, count_include_pad=False), 
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool1d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine), 
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine), 
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv1d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv1d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
}


class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm1d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)


class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv1d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm1d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv1d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm1d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv1d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm1d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

'''
class Zero(nn.Module):
  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride
  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)
'''

#x = torch.rand(2,64,10)
# stride=2
# zero = Zero(2)
# out = zero(x)
# out.shape


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride
  def forward(self, x):
    n, c, h = x.size()
    h /= self.stride # divides h with stride and rounds result off (abrunden)
    h = math.ceil(h)
    if x.is_cuda:
      with torch.cuda.device(x.get_device()):
        padding = torch.cuda.FloatTensor(n, c, h).fill_(0)
    else:
      padding = torch.FloatTensor(n, c, h).fill_(0)
    return padding    


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv1d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) # jede output channel wird durch 2 geteilt, aber da am ende zusammengefügt wird, hat output C_out als output channels
    self.conv_2 = nn.Conv1d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm1d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    #print(self.conv_1(x).shape) 
    #print(self.conv_2(x[:,:,1:]).shape)

    out = torch.cat([self.conv_1(x), self.conv_2(x)], dim=1) # anzahl batches und anzahl channels soll gleich bleiben , aber im 
    # 2d fall soll die von beiden neuronen dimensinonen die erste spalte bzw. erste zeile gekickt werden
    # an dimension 1 zusammenfügten, also anhand von channels (weil batches ist dimension 0) zusammenfügen
    #out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1) 
    out = self.bn(out)
    return out
    
    

