import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

def np2tensor(img):
    """
    img: BS x [C x] x H x W
    """
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    
    if img.ndim == 3:
        img = img.unsqueeze(1)
    elif img.ndim == 4:
        pass
    else:
        raise TypeError(f"Invalid ndim of img: {img.ndim}")
    
    if img.size(-1) == 3:
        img = img.permute(0, 3, 1, 2)

    return img

def tensor2np(img: Tuple[np.ndarray, Tensor]):
    """
    img: BS x [C x] x H x W
    """
    if isinstance(img, Tensor):
        if img.requires_grad:
            img = img.detach()
        img = img.cpu().numpy()
    elif isinstance(img, np.ndarray):
        pass
    else:
        raise TypeError(f"Wrong input type: {type(img)}!")

    return img

def gkern(klen, nsig):
    """
    Returns a Gaussian kernel array. Convolution with it results in image blurring.
    """
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

def gaussian_blur(img, klen=11, ksig=5):
    # get gkern
    kern = gkern(klen, ksig)
    # compute gaussian blur
    img_out = nn.functional.conv2d(img, kern, padding=klen//2)
    return img_out

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class CausalMetric():
    def __init__(self, model, mode, step):
        r"""Create deletion/insertion metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step

    def single_run(self, img_tensor, explanation, verbose=False, path = "abc"):
        r"""Run metric on one image-saliency pair.

        Args:
            img_tensor (Tensor): normalized image tensor.
            explanation (np.ndarray): saliency map.
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - also plot final step.
                2 - also plot every step and print 2 top classes.
            save_to (str): directory to save every step plots to.

        Return:
            scores (nd.array): Array containing scores at every step.
        """
        _, _, H, W = img_tensor.shape
        HW = H * W # image area
        explanation = np.array(explanation)
        pred = self.model(img_tensor)
        _, c = torch.max(pred, 1)
        c = c.cpu().numpy()[0]
        n_steps = (HW + self.step - 1) // self.step

        if self.mode == 'del':
            title = 'Deletion game'
            ylabel = 'Pixels deleted'
            start = img_tensor.clone()
            finish = torch.zeros_like(img_tensor)

        elif self.mode == 'ins':
            title = 'Insertion game'
            ylabel = 'Pixels inserted'
            start = gaussian_blur(img_tensor)
            finish = img_tensor.clone()

        scores = np.empty(n_steps + 1)
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(explanation.reshape(-1, HW), axis=1), axis=-1)
        for i in range(n_steps+1):
            pred = self.model(start)
            pred = torch.softmax(pred, dim=1)
            scores[i] = pred[0, c]

            # Render image if verbose, if it's the last step.
            if verbose == True and i == n_steps:
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, scores[i]))
                plt.axis('off')
                plt.imshow(start[0].permute(1,2,0))

                plt.subplot(122)
                plt.plot(np.arange(i+1) / n_steps, scores[:i+1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i+1) / n_steps, 0, scores[:i+1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                plt.savefig(path)
                plt.show()

            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                start.cpu().numpy().reshape(1, 3, HW)[0, :, coords] = finish.cpu().numpy().reshape(1, 3, HW)[0, :, coords]
        return auc(scores), c
    