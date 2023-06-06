import cv2
import torch

import numpy as np
from itertools import islice
from tqdm import tqdm
from torchvision import transforms

from models.clipseg import CLIPDensePredT

model = CLIPDensePredT(version="ViT-B/16", reduce_dim=64)
model.eval()
model.load_state_dict(torch.load("weights/rd64-uni.pth", map_location=torch.device('cpu')), strict=False)

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]), 
    transforms.Resize((352, 352), antialias=True)
    # transforms.Resize((960, 960), antialias=True)
])

        
class VideoFrameLoader:
    def __init__(self, filename, batch_size):
        self.video_frame_iter = self.__VideoFrameIter(filename)
        self.batch_size = batch_size

    def __iter__(self):
        return self
    
    def __next__(self):
        frames = list(islice(self.video_frame_iter, batch_size))

        if len(frames) == 0:
            raise StopIteration
        else:
            return np.concatenate([np.expand_dims(frame, 0) 
                                   for frame in frames])
            
    def get_fps(self):
        return self.video_frame_iter.get_fps()

    class __VideoFrameIter:
        def __init__(self, filename):
            self.filename = filename
            self.video_capture = cv2.VideoCapture(filename)

        def __iter__(self):
            return self
        
        def __next__(self):
            is_success, frame = self.video_capture.read()

            if not is_success:
                self.video_capture.release()
                raise StopIteration
            else:
                return frame.copy()

        def get_fps(self):
            return self.video_capture.get(cv2.CAP_PROP_FPS)

batch_size = 32
input_frames = VideoFrameLoader("video-1-trim.mp4", batch_size)
output_frames = []
input_fps = input_frames.get_fps()
output_fps = input_fps
for i, batch in enumerate(tqdm(input_frames)):
    inp = torch.cat([transform(_).unsqueeze(0) for _ in batch])
    prompt = ["woman with orange t-shirts"] * len(batch)

    with torch.no_grad():
        pred, *_ = model(inp, prompt)

    # masks = transforms.Resize(batch[0].shape[:2], antialias=True)(torch.sigmoid(pred)*.7 + .3).squeeze(1)
    masks = transforms.Resize(batch[0].shape[:2], antialias=True)(torch.where(torch.sigmoid(pred) ** 2 * 10 + .1 > 1., 1., torch.sigmoid(pred) ** 2 * 10 + .1)).squeeze(1)
    masks_numpy = masks.numpy()

    for inp_frame, mask in zip(batch, masks_numpy):
        frame = cv2.cvtColor(inp_frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        frame[:, :, 2] *= mask
        frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_HSV2BGR)

        output_frames.append(frame)

out = cv2.VideoWriter("woman-orange-tshirt.avi",cv2.VideoWriter_fourcc(*'MJPG'), output_fps, output_frames[0].shape[:2][::-1])
if not out.isOpened():
    print("not opened")
else:
    for frame in output_frames:
        out.write(frame)
    out.release()