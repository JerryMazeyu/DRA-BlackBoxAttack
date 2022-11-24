import os
import cv2
import torch
import logging

def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def save_images(input_tensors, filenames, output_dir):
    assert (len(input_tensors.shape) == 4)
    for i in range(input_tensors.shape[0]):
        input_tensor = input_tensors[i,:,:,:]
        filename = filenames[i]
        input_tensor = input_tensor.clone().detach()
        input_tensor = input_tensor.to(torch.device('cpu'))
        input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
        input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, filename), input_tensor)

def lprint(*args, dir=None):
    logger = logging.getLogger("Running log")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    if dir:
        fh = logging.FileHandler(os.path.join(dir, "Log.log"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    args = [str(x) for x in args]
    msg = ' '.join(args)
    logger.info(msg)



    


