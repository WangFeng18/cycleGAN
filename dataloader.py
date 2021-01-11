import os
import glob
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class unlabledDataset(Dataset):
	def __init__(self, root, transforms=None):
		self.imglist = glob.glob(os.path.join(root, '*.jpg'))
		self.transforms = transforms

	def __getitem__(self, i):
		path = self.imglist[i]
		with open(path, 'rb') as f:
			img = Image.open(f)
			img = img.convert('RGB')
		if self.transforms is not None:
			img = self.transforms(img)
		return img

	def __len__(self):
		return len(self.imglist)

def get_loader(root, args):
	aug = transforms.Compose([
		transforms.Resize(286, Image.BICUBIC),
		transforms.RandomCrop(256),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])
	dataset = unlabledDataset(root, transforms=aug)
	loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=args.n_workers)	
	return loader
