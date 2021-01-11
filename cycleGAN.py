import os
from tqdm import tqdm
import argparse
import logging
import torch
import torchvision
import torch.nn as nn
import numpy as np
import itertools
from dataloader import get_loader
from model import ResnetGenerator as generator
from model import NLayerDiscriminator as discriminator
from model import ImagePool
from model import init_weights
from model import LSGANLoss
from utils import *
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

class cycleGAN(object):
	def __init__(self, args):
		self.args = args 
		self.device = torch.device('cuda: 0')
		self.device_ids = list(map(lambda x: int(x), args.gpus.split(',')))
		self.loader1 = get_loader(args.root1, args)
		self.loader2 = get_loader(args.root2, args)
		self.netG_A = generator(input_nc=3, output_nc=3, n_blocks=9)
		self.netG_B = generator(input_nc=3, output_nc=3, n_blocks=9)
		self.netD_A = discriminator(input_nc=3)
		self.netD_B = discriminator(input_nc=3)
		self.netG_A.to(self.device)
		self.netG_B.to(self.device)
		self.netD_A.to(self.device)
		self.netD_B.to(self.device)

		self.netG_A = nn.DataParallel(self.netG_A, device_ids=self.device_ids)
		self.netG_B = nn.DataParallel(self.netG_B, device_ids=self.device_ids)
		self.netD_A = nn.DataParallel(self.netD_A, device_ids=self.device_ids)
		self.netD_B = nn.DataParallel(self.netD_B, device_ids=self.device_ids)
		init_weights(self.netG_A)
		init_weights(self.netG_B)
		init_weights(self.netD_A)
		init_weights(self.netD_B)
		
		self.fake_A_pool = ImagePool(50)
		self.fake_B_pool = ImagePool(50)
		self.criterionCycle = nn.L1Loss()
		self.criterionGAN = LSGANLoss()
		self.start_epoch = 0

		self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=self.args.lr, betas=(0.5, 0.999))
		self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=self.args.lr, betas=(0.5, 0.999))

		def lambda_rule(epoch):
			lr_l = 1.0 - max(0, epoch + 1 - 100) / float(100 + 1)
			return lr_l
		self.scheduler_G = lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lambda_rule)
		self.scheduler_D = lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=lambda_rule)
		self.create_experiment()
		self.get_logger()

	def create_experiment(self):
		args = self.args
		if not os.path.exists(args.exp):
			os.makedirs(args.exp)
		if not os.path.exists(os.path.join(args.exp, 'models')):
			os.makedirs(os.path.join(args.exp, 'models'))
		if not os.path.exists(os.path.join(args.exp, 'samples')):
			os.makedirs(os.path.join(args.exp, 'samples'))

	def get_logger(self):
		args = self.args
		if not os.path.exists(os.path.join(args.exp, 'logs')):
			os.makedirs(os.path.join(args.exp, 'logs'))
		if not os.path.exists(os.path.join(args.exp, 'runs')):
			os.makedirs(os.path.join(args.exp, 'runs'))
		self.writer = SummaryWriter(logdir=os.path.join(self.args.exp, 'runs'))
		self.logger = getLogger(args.exp)

	def set_discriminator_grads(self, active):
		for param in self.netD_A.parameters():
			param.requires_grad = active
		for param in self.netD_B.parameters():
			param.requires_grad = active

	def load_checkpoint(self, path):
		args = self.args
		logging.info('Resume from {}'.format(args.resume_path))
		checkpoint = torch.load(args.resume_path)
		self.netG_A.load_state_dict(checkpoint['netG_A'])
		self.netG_B.load_state_dict(checkpoint['netG_B'])
		self.netD_A.load_state_dict(checkpoint['netD_A'])
		self.netD_B.load_state_dict(checkpoint['netD_B'])

		self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
		self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
		self.start_epoch = checkpoint['epoch']

	def main_loop(self):
		if self.args.resume_path != '':
			self.load_checkpoint(self.args.resume_path)
		avgmeters = AvgMeters()
		itA, itB = iter(self.loader1), iter(self.loader2)
		for epoch in range(self.start_epoch, self.args.max_epoch):
			# train cycleGAN for one epoch (the epoch is from modal A)
			self.scheduler_G.step()
			self.scheduler_D.step()
			pbar = tqdm(range(len(self.loader1.dataset)))
			for i in pbar:
				# fetch data
				try:
					realA, realB = next(itA).to(self.device), next(itB).to(self.device)
				except StopIteration:
					itA, itB = iter(self.loader1), iter(self.loader2)
					realA, realB = next(itA).to(self.device), next(itB).to(self.device)
				fakeB, fakeA = self.netG_A(realA), self.netG_B(realB)
				recA,  recB  = self.netG_B(fakeB), self.netG_A(fakeA)

				if np.random.rand() < 0.01:
					display = torchvision.utils.make_grid([realA[0].detach().cpu(), fakeB[0].detach().cpu(), recA[0].detach().cpu(), realB[0].detach().cpu(), fakeA[0].detach().cpu(), recB[0].detach().cpu()], nrow=3, normalize=True, scale_each=True)
					self.writer.add_image('Image_Epoch{}_Iter{}'.format(epoch, i), display, epoch)

				self.set_discriminator_grads(active=False)
				loss_gan_a = self.criterionGAN(self.netD_A(fakeB).to(self.device), 1.0)
				loss_gan_b = self.criterionGAN(self.netD_B(fakeA).to(self.device), 1.0)
				loss_cyc_a = self.criterionCycle(recA.to(self.device), realA) * 10.
				loss_cyc_b = self.criterionCycle(recB.to(self.device), realB) * 10.
				loss_G = loss_gan_a +loss_gan_b + loss_cyc_a + loss_cyc_b
				self.optimizer_G.zero_grad()
				loss_G.backward()
				self.optimizer_G.step()
				avgmeters.add('gan_a', loss_gan_a.item())
				avgmeters.add('gan_b', loss_gan_b.item())
				avgmeters.add('cyc_a', loss_cyc_a.item())
				avgmeters.add('cyc_b', loss_cyc_b.item())
				avgmeters.add('G', loss_G.item())
				self.set_discriminator_grads(active=True)

				fakeB = self.fake_B_pool.query(fakeB.detach())
				fakeA = self.fake_A_pool.query(fakeA.detach())
				loss_D_A = 0.5 * self.criterionGAN(self.netD_A(realB).to(self.device), 1.0) + 0.5 * self.criterionGAN(self.netD_A(fakeB).to(self.device), 0.0)
				loss_D_B = 0.5 * self.criterionGAN(self.netD_B(realA).to(self.device), 1.0) + 0.5 * self.criterionGAN(self.netD_B(fakeA).to(self.device), 0.0)
				loss_D = loss_D_A + loss_D_B
				self.optimizer_D.zero_grad()
				loss_D.backward()
				self.optimizer_D.step()
				avgmeters.add('D_A', loss_D_A.item())
				avgmeters.add('D_B', loss_D_B.item())
				avgmeters.add('D', loss_D.item())

				lr = self.optimizer_G.param_groups[0]['lr']
				pbar.set_description("Epoch:{} [lr:{}]".format(epoch, lr))
				info = 'gan_a:{:.4f}|gan_b:{:.4f}|cyc_a:{:.4f}|cyc_b:{:.4f}|G:{:.4f}|D_A:{:.4f}|D_B:{:.4f}|D:{:.4f}'.format(avgmeters.get('gan_a'), avgmeters.get('gan_b'), avgmeters.get('cyc_a'), avgmeters.get('cyc_b'), avgmeters.get('G'), avgmeters.get('D_A'), avgmeters.get('D_B'), avgmeters.get('D'))
				pbar.set_postfix(info=info)
			# record and visualization
			checkpoint = {
				'epoch': epoch + 1,
				'netG_A': self.netG_A.state_dict(),
				'netG_B': self.netG_B.state_dict(),
				'netD_A': self.netD_A.state_dict(),
				'netD_B': self.netD_B.state_dict(),
				'optimizer_G': self.optimizer_G.state_dict(),
				'optimizer_D': self.optimizer_D.state_dict(),
			}
			torch.save(checkpoint, os.path.join(self.args.exp, 'models', 'checkpoint.pth'))
			logging.info('Epoch: {}'.format(epoch))
			for k in ['gan_a', 'gan_b', 'cyc_a', 'cyc_b', 'G', 'D_A', 'D_B', 'D']:
				logging.info('\t {}: {:.4f}'.format(k, avgmeters.get(k)))
				self.writer.add_scalar(k, avgmeters.get(k), epoch)
			avgmeters.clear()

				


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--root1', default='/home/user/monet2photo/trainA/', type=str)
	parser.add_argument('--root2', default='/home/user/monet2photo/trainB/', type=str)
	parser.add_argument('--gpus', default='0,1,2,3', type=str)
	parser.add_argument('--max_epoch', default=201, type=int)
	parser.add_argument('--exp', default='/matrix/cycleGAN/monet2photo/', type=str)
	parser.add_argument('--resume_path', default='', type=str)
	parser.add_argument('--lr', default=0.0002, type=float)
	parser.add_argument('--batch_size', default=1, type=int)
	parser.add_argument('--n_workers', default=32, type=int)
	# exclusive best to be 0
	args = parser.parse_args()
	runner = cycleGAN(args)
	runner.main_loop()
