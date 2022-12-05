
import numpy as np
import torchvision.transforms as transforms

def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim
    if patch_size > 1:
        batch_size = np.shape(img_tensor)[0]
        seq_length = np.shape(img_tensor)[1]
        img_height = np.shape(img_tensor)[2]
        img_width = np.shape(img_tensor)[3]
        num_channels = np.shape(img_tensor)[4]
        a = np.reshape(img_tensor, [batch_size, seq_length,
                                    img_height//patch_size, patch_size,
                                    img_width//patch_size, patch_size,
                                    num_channels])
        b = np.transpose(a, [0,1,2,4,3,5,6])
        patch_tensor = np.reshape(b, [batch_size, seq_length,
                                    img_height//patch_size,
                                    img_width//patch_size,
                                    patch_size*patch_size*num_channels])
        
        return patch_tensor
    else:
        return img_tensor

def reshape_patch_back(img_tensor, patch_size):
    assert 5 == img_tensor.ndim
    if patch_size > 1:
        batch_size = np.shape(img_tensor)[0]
        seq_length = np.shape(img_tensor)[1]
        patch_height = np.shape(img_tensor)[2]
        patch_width = np.shape(img_tensor)[3]
        channels = np.shape(img_tensor)[4]
        img_channels = channels // (patch_size*patch_size)
        a = np.reshape(img_tensor, [batch_size, seq_length,
                                    patch_height, patch_width,
                                    patch_size, patch_size,
                                    img_channels])
        b = np.transpose(a, [0,1,2,4,3,5,6])
        patch_tensor = np.reshape(b, [batch_size, seq_length,
                                    patch_height * patch_size,
                                    patch_width * patch_size,
                                    img_channels])
        return patch_tensor
    else:
        return img_tensor