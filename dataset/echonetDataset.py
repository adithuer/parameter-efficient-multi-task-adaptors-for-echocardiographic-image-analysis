import numpy as np
import pims
import skimage.draw
import pandas as pd
import torch

from tqdm import tqdm

class EchoNetDataset():
    def __init__(self, type, n_classes, path_filelist, path_volumelist, path_video, transform=None, transform_data=None, transform_target=None):
        self.n_classes = n_classes

        df_samples = self.get_data_info(type, path_filelist, path_volumelist)

        data = []
        masks = []

        for _, sample in tqdm(df_samples.iterrows()):
            img, mask = self.preprocess(sample, path_video)
            data.append(img)
            masks.append(mask)

        self.data = torch.Tensor(np.array(data) / 255.0).permute((0,3,1,2))
        self.target_masks = torch.nn.functional.one_hot(torch.Tensor(np.array(masks)).long(), self.n_classes).float().permute((0,3,1,2))
        
        self.transform = transform
        self.transform_data = transform_data
        self.transform_target = transform_target

    def get_mask(self, df_sample, width, height):
        x1, x2 = np.array(df_sample['X1']), np.array(df_sample['X2'])
        y1, y2 = np.array(df_sample['Y1']), np.array(df_sample['Y2'])
        x = np.concatenate((x1[1:], np.flip(x2[1:])))
        y = np.concatenate((y1[1:], np.flip(y2[1:])))

        r, c = skimage.draw.polygon(np.rint(y).astype(int), np.rint(x).astype(int), (width, height))
        mask = np.zeros((width, height), np.float32)
        mask[r, c] = 1

        return mask
    
    
    def preprocess(self, df_sample, path_video):
        video_name = df_sample['FileName']
        frame_id = df_sample['Frame']

        v = pims.Video(path_video + '/' + video_name)
        frame = np.array(v)[frame_id]
        width, height, _ = frame.shape
        
        mask = self.get_mask(df_sample, width, height)

        return np.array(frame), np.array(mask)
    
    
    def get_data_info(self, type, path_filelist, path_volumelist):
        # load video information
        df = pd.read_csv(path_filelist).dropna(axis=0, how='any')
        df['FileName'] = df['FileName'] + '.avi'
        df = df[df['Split'] == type].drop(columns=['FrameHeight','FrameWidth','FPS','NumberOfFrames'])

        # load volume information
        df_volume = pd.read_csv(path_volumelist).dropna(axis=0, how='any')
        df_volume = df_volume.groupby(['FileName', 'Frame']).agg(list).reset_index()

        df = pd.merge(df, df_volume, on='FileName').sort_values(['FileName', 'Frame'])

        # replace ESV and EDV with EV
        df['EV'] = df['ESV']
        df.iloc[::2]['EV'].values[:] = df.iloc[::2]['EDV'].values[:]
        df_samples = df.drop(columns=['ESV', 'EDV'])

        df_samples = df_samples.iloc[:10]

        return df_samples
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample, label = self.data[index], self.target_masks[index]

        # preprocess image
        if self.transform_data:
            sample = self.transform_data(sample)
        
        # preprocess label
        if self.transform_target:
            label = self.transform_target(label)

        # transform image and label
        if self.transform:
            sample = torch.cat([sample, label], dim=0)
            sample = self.transform(sample)
            sample, label = torch.split(sample, [sample.shape[0] - self.n_classes, self.n_classes], dim=0)

        return sample, label