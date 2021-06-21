import argparse
import torch
import torchvision
import torchvision.utils as utils
import torch.nn as nn
from random import randint
from model import NetD, NetG

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--imageSize', type=int, default=96)
parser.add_argument('--nz', type=int, default=256, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='size of G')
parser.add_argument('--ndf', type=int, default=64, help='size of D')
parser.add_argument('--epoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--data_dir', default='data/', help='folder to train data')
parser.add_argument('--out_dir', default='imgs/', help='Results')
opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Scale(opt.imageSize),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

dataset = torchvision.datasets.ImageFolder(opt.data_dir, transform=transforms)

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    drop_last=True,
)

netG = NetG(opt.ngf, opt.nz).to(device)
netD = NetD(opt.ndf).to(device)

criterion = nn.BCELoss()
optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

for epoch in range(1, opt.epoch + 1):
    for i, (imgs,_) in enumerate(dataloader):
        optimizerD.zero_grad()
        imgs = imgs.to(device)
        output = netD(imgs)
        output = torch.squeeze(output)
        label.data.fill_(real_label)
        label = label.to(device)

        errD_real = criterion(output, label)
        errD_real.backward()
        label.data.fill_(fake_label)
        noise = torch.randn(opt.batchSize, opt.nz, 1, 1)
        noise = noise.to(device)
        fake = netG(noise)
        output = netD(fake.detach())
        output = torch.squeeze(output)

        errD_fake = criterion(output, label)
        errD_fake.backward()
        errD = (errD_fake + errD_real) / 2
        optimizerD.step()

        optimizerG.zero_grad()

        label.data.fill_(real_label)
        label = label.to(device)
        output = netD(fake)
        output=torch.squeeze(output)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
              % (epoch, opt.epoch, i, len(dataloader), errD.item(), errG.item()))

    utils.save_image(fake.data,
                      '%s/fake_samples_epoch_%03d.png' % (opt.out_dir, epoch),
                      normalize=True)


