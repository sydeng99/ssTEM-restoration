import torch
from PIL import Image
from skimage import io
import numpy as np
from torch.autograd import Variable

def Gray2Tensor(im):
    im=(im / 255.).astype('float32')
    im = im[np.newaxis, np.newaxis, :, :, ]
    tensor = Variable(torch.from_numpy(im))
    tensor=tensor.cuda()
    return tensor

def Tensor2Gray(tensor):
    tensor = tensor.cpu()
    [b, c, w, h] = tensor.shape
    tensor = tensor.data.numpy()
    im_out = tensor[0, 0, :, :] * 255.
    im_out = im_out.astype('uint8')
    return im_out

def TrainTensor2Gray(tensor):
    im=(np.squeeze(tensor[0][0].data.cpu().numpy()) * 255).astype(np.uint8)
    return im

def TrainTensor2mask(tensor):
    pred_denoise1_mask = np.squeeze(tensor[0, 0].data.cpu().numpy())
    pred_denoise1_mask[pred_denoise1_mask > 1] = 1
    pred_denoise1_mask[pred_denoise1_mask < 0] = 0
    pred_denoise1_mask = (pred_denoise1_mask * 255).astype(np.uint8)
    return pred_denoise1_mask