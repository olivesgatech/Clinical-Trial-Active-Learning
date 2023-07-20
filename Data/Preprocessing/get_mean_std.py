import pandas as pd
import numpy as np
import torch
from torchvision.transforms import transforms

from dataset_v2 import GetDataset
from PIL import ImageStat

class Stats(ImageStat.Stat):
  def __add__(self, other):
    # add self.h and other.h element-wise
    return Stats(list(np.add(self.h, other.h)))

spreadsheet_path = '/home/byeolyi/activelearning/Spreadsheets/'
tr_path = spreadsheet_path + 'prime_trex_compressed.csv'
data = pd.read_csv(tr_path)



loader = torch.utils.data.DataLoader(
    GetDataset(data, '/data/Datasets/'),
    batch_size=len(data),
    num_workers=0,
    shuffle=False
)
def mean_std(loader):
    images = next(iter(loader))
    print(images.shape)
    # shape of images = [batch_size,width,height]
    mean, std = images.mean([0, 1, 2]), images.std([0, 1, 2])

    return mean, std
mean, std = mean_std(loader)
print('mean: ', str(mean))
print('std: ', str(std))
# statistics = None
# toPIL=transforms.ToPILImage()
#
#
# for data in loader:
#     for b in range(data.shape[0]):
#         if statistics is None:
#             print(type(toPIL(data[b])))
#             statistics = Stats(toPIL(data[b]))
#         else:
#             statistics += Stats(toPIL(data[b]))
# print(f'mean:{statistics.mean}, std:{statistics.stddev}')



# mean = 0.
# meansq = 0.
# count = 0.
# for data in loader:
#     mean = data.sum()
#     meansq = meansq + (data**2).sum()
#     count += np.prod(data.shape)
# mean = mean/count
# print("mean: " + str(mean))
# total_var = (meansq/count) - (mean**2)
# std = torch.sqrt(total_var)
#
# # var = 0.0
# # for images in loader:
# #     batch_samples = images.size(0)
# #     images = images.view(batch_samples, images.size(1), -1)
# #     var += ((images - mean.unsqueeze(1))**2).sum([0,2])
# # std = torch.sqrt(var / (len(loader.dataset)*128*128))
#
# #std = torch.sqrt(meansq - mean ** 2)
# print("std: " + str(std))
# print()