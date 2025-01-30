import torch
import torch.nn as nn

import pandas as pd
from pathlib import Path
import torch
from typing import List, Tuple
from torchvision.transforms import v2

from tqdm import tqdm
from dataset.camusHelpers import *


class CAMUSDataset():
    def __init__(self, *, 
                 type: str, 
                 n_classes: int, 
                 n_views: int, 
                 path_splitInfo: str, 
                 path_patients: str, 
                 transform=None, 
                 transform_data=None, 
                 transform_target=None, 
                 return_value: str='mask', 
                 no_poor: bool=False) -> None:
        self.transform = transform
        self.transform_data = transform_data
        self.transform_target = transform_target
        self.return_value = return_value
        self.n_classes = n_classes
              
        patients_dir = Path(path_patients)
        df = self.get_patient_infos(patients_dir)

        if no_poor:
            df = df[df['ImageQuality'] != 'Poor']

        if type == 'TRAIN':
            path_splitInfo = path_splitInfo + '/subgroup_training.txt'
        elif type == 'VAL':
            path_splitInfo = path_splitInfo + '/subgroup_validation.txt'
        elif type == 'TEST':
            path_splitInfo = path_splitInfo + '/subgroup_testing.txt'
        else:
            raise Exception("No valid training type")

        df_splits = pd.read_csv(path_splitInfo, header=None, names=['Name'])
        df = pd.merge(df, df_splits, on='Name')
        df = df.dropna(subset=['EF','Name','Age','ED','ES'])

        data, masks, volumes, views, ages = self.preprocess(df, patients_dir)
        self.data = torch.stack(data, dim=0)

        if self.return_value == 'age':
             self.target = torch.Tensor(np.array(ages)).unsqueeze(dim=-1)
        elif self.return_value == 'view':
            self.target = nn.functional.one_hot(torch.Tensor(np.array(views)).long(), n_views).float()
        elif self.return_value == 'volume':
            self.target = torch.Tensor(np.array(volumes)).unsqueeze(dim=-1)
        else:
            self.target = nn.functional.one_hot(torch.stack(masks, dim=0).long(), self.n_classes).float().permute((0,3,1,2))
        

    def preprocess(self, df: pd.DataFrame, patient_dir: str, lv_label: int =1, ef_thresh: int=3, size: int=512) -> List[list, list, list, list]:
        data = []
        masks = []
        volumes = []
        views = []
        ages = []

        for _, patient in tqdm(df.iterrows(), desc=f"Loading patient images"):
            patient_name = patient['Name']

            mask_pattern = "{patient_name}/{patient_name}_{view}_{instant}_gt.nii.gz"
            img_pattern = "{patient_name}/{patient_name}_{view}_{instant}.nii.gz"

            img_a2c_ed, _ = sitk_load(patient_dir / img_pattern.format(patient_name=patient_name, view="2CH", instant="ED"))
            img_a2c_es, _ = sitk_load(patient_dir / img_pattern.format(patient_name=patient_name, view="2CH", instant="ES"))
            img_a4c_ed, _ = sitk_load(patient_dir / img_pattern.format(patient_name=patient_name, view="4CH", instant="ED"))
            img_a4c_es, _ = sitk_load(patient_dir / img_pattern.format(patient_name=patient_name, view="4CH", instant="ES"))

            a2c_ed, a2c_info = sitk_load(patient_dir / mask_pattern.format(patient_name=patient_name, view="2CH", instant="ED"))
            a2c_es, _ = sitk_load(patient_dir / mask_pattern.format(patient_name=patient_name, view="2CH", instant="ES"))
            a2c_voxelspacing = a2c_info["spacing"][:2][::-1]

            a4c_ed, a4c_info = sitk_load(patient_dir / mask_pattern.format(patient_name=patient_name, view="4CH", instant="ED"))
            a4c_es, _ = sitk_load(patient_dir / mask_pattern.format(patient_name=patient_name, view="4CH", instant="ES"))
            a4c_voxelspacing = a4c_info["spacing"][:2][::-1]

            a2c_ed_lv_mask = a2c_ed == lv_label
            a2c_es_lv_mask = a2c_es == lv_label
            a4c_ed_lv_mask = a4c_ed == lv_label
            a4c_es_lv_mask = a4c_es == lv_label

            edv, esv = compute_left_ventricle_volumes(a2c_ed_lv_mask, a2c_es_lv_mask, a2c_voxelspacing, a4c_ed_lv_mask, a4c_es_lv_mask, a4c_voxelspacing)
            ef = round(100 * (edv - esv) / edv)


            # crop images to same length ratios
            height, _ = img_a2c_ed.shape
            img_a2c_ed = v2.CenterCrop(size=height)(torch.Tensor(img_a2c_ed))
            a2c_ed_lv_mask = v2.CenterCrop(size=height)(torch.Tensor(a2c_ed_lv_mask))

            height, _ = img_a2c_es.shape
            img_a2c_es = v2.CenterCrop(size=height)(torch.Tensor(img_a2c_es))
            a2c_es_lv_mask = v2.CenterCrop(size=height)(torch.Tensor(a2c_es_lv_mask))

            height, _ = img_a4c_ed.shape
            img_a4c_ed = v2.CenterCrop(size=height)(torch.Tensor(img_a4c_ed))
            a4c_ed_lv_mask = v2.CenterCrop(size=height)(torch.Tensor(a4c_ed_lv_mask))

            height, _ = img_a4c_es.shape
            img_a4c_es = v2.CenterCrop(size=height)(torch.Tensor(img_a4c_es))
            a4c_es_lv_mask = v2.CenterCrop(size=height)(torch.Tensor(a4c_es_lv_mask))


            # resize images to same size
            a4c_ed_lv_mask = v2.Resize(size=(size))(a4c_ed_lv_mask.unsqueeze(dim=0)).squeeze(dim=0)
            a4c_es_lv_mask = v2.Resize(size=(size))(a4c_es_lv_mask.unsqueeze(dim=0)).squeeze(dim=0)
            a2c_ed_lv_mask = v2.Resize(size=(size))(a2c_ed_lv_mask.unsqueeze(dim=0)).squeeze(dim=0)
            a2c_es_lv_mask = v2.Resize(size=(size))(a2c_es_lv_mask.unsqueeze(dim=0)).squeeze(dim=0)

            img_a2c_ed = v2.Resize(size=(size))(img_a2c_ed.unsqueeze(dim=0))
            img_a2c_es = v2.Resize(size=(size))(img_a2c_es.unsqueeze(dim=0))
            img_a4c_ed = v2.Resize(size=(size))(img_a4c_ed.unsqueeze(dim=0))
            img_a4c_es = v2.Resize(size=(size))(img_a4c_es.unsqueeze(dim=0))

            if abs(ef - patient['EF']) > ef_thresh:
                print(f'For {patient_name} the calculated EF differs from the measured EF more than {ef_thresh}')
            else:
                data.extend((img_a4c_ed, img_a4c_es, img_a2c_ed, img_a2c_es))
                masks.extend((a4c_ed_lv_mask, a4c_es_lv_mask, a2c_ed_lv_mask, a2c_es_lv_mask))
                volumes.extend((edv, esv, edv, esv))
                views.extend((1, 1, 0, 0))

                age = patient['Age']
                ages.extend((age, age, age, age))


        return data, masks, volumes, views, ages

    def get_patient_infos(self, patients_dir: str) -> pd.DataFrame:
        df = pd.DataFrame(columns=['Name', 'ED', 'NbFrame', 'Sex', 'Age', 'ImageQuality', 'EF', 'FrameRate'])        

        for patient_dir in tqdm(patients_dir.iterdir(), desc="Loading patient information"):
            patient = {'Name': patient_dir.name}

            with open(patient_dir / 'Info_2CH.cfg') as file:
                for line in file:
                    key, value = line.split(": ")
                    value = value.replace('\n', '')

                    patient[key] = value

            df = df._append(patient, ignore_index = True)

        df['ED'] = pd.to_numeric(df['ED'])
        df['NbFrame'] = pd.to_numeric(df['NbFrame'])
        df['Age'] = pd.to_numeric(df['Age'])
        df['EF'] = pd.to_numeric(df['EF'])
        df['FrameRate'] = pd.to_numeric(df['FrameRate'])
        df['ES'] = pd.to_numeric(df['ES'])
        
        return df
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        sample, label = self.data[index], self.target[index]

        if self.transform_data:
            sample = self.transform_data(sample)
        
        if self.transform_target and self.return_value == 'mask':
            label = self.transform_target(label)

        # transform image and label
        if self.transform:
            sample = torch.cat([sample, label], dim=0)
            sample = self.transform(sample)
            sample, label = torch.split(sample, [sample.shape[0] - self.n_classes, self.n_classes], dim=0)

        return sample, label