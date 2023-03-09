from PIL import Image
import torch

class LoadDataSet():
    def __init__(self,fids,flabels,path,transforms) -> None:
        self.fids = fids
        self.fnames = [path+fid+".jpeg" for fid in self.fids]
        self.labels = flabels
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        # print(self.fnames[idx])
        img = Image.open(self.fnames[idx]).convert("RGB")
        img_t = self.transforms(img)
        # print(img_t.shape)
        return img_t,self.labels[idx]

class TestDataset():
    def __init__(self,fids,path,transforms) -> None:
        self.fids = fids
        self.fnames = [path + fid + ".jpeg" for fid in self.fids]
        self.transforms = transforms

    def __len__(self):
        return len(self.fids)

    def __getitem__(self,idx):
        img = Image.open(self.fnames[idx]).convert("RGB")
        img_t = self.transforms(img)
        return img_t,self.fids[idx]
