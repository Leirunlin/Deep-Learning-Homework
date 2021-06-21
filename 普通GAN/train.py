import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from utils import *
from torchvision.utils import save_image
from torch.autograd import Variable


def default_loader(path):
    return Image.open(path).convert('RGB')


class Custom_Dataset(Dataset):
    def __init__(self, img_path, loader=default_loader):
        self.imgs = []
        for root, dir, files in os.walk(img_path):
            for file in files:
                self.imgs.append(root + "/" + file)
        self.loader = loader
        self.transform = transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )
        self.batch_size = opt.batch_size

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.FloatTensor

# ----------
#  Training
# ----------

dataloader = Custom_Dataset("./dataset/mix")
dataloader = DataLoader(dataset=dataloader, batch_size=opt.batch_size, shuffle=True)

for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(dataloader):
        valid = Variable(torch.ones(size=(imgs.size(0), 1)), requires_grad=False)
        fake = Variable(torch.zeros(size=(imgs.size(0), 1)), requires_grad=False)
        real_imgs = Variable(imgs.type(Tensor))
        optimizer_G.zero_grad()
        z = Variable(torch.randn(size=(imgs.shape[0], opt.latent_dim)))
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25].reshape(real_imgs.data[:25].shape), "images/%d.png" % batches_done, nrow=5, normalize=True)
            save_image(real_imgs.data[:25], "images/real_%d.png" % batches_done, nrow=5, normalize=True)