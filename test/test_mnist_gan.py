# rough copy of https://github.com/geohot/tinygrad/blob/master/examples/mnist_gan.py

from simplegrad import Tensor, Device, Adam
import numpy as np
import itertools as it
from torchvision.utils import make_grid, save_image
import torch
from abc import abstractmethod
import os

def leakyrelu(x, neg_slope=0.2):
    return x.relu().sub(x.fork().mul(Tensor(neg_slope).mul(Tensor(-1.0))).relu())
    torch.functional.F.leaky_relu(torch.tensor(x.val), negative_slope=0.2)

def random_uniform(*shape):
    return np.random.uniform(-1., 1., size=shape)/np.sqrt(np.prod(shape)).astype(np.float32)

class nn:
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @property
    def params(self):
        return tuple(v for k,v in self.__dict__.items() if isinstance(v, Tensor))

class LinearGen(nn):
    def __init__(self):
        self.l1 = Tensor(random_uniform(128,256))
        self.l2 = Tensor(random_uniform(256, 512))
        self.l3 = Tensor(random_uniform(512, 1024))
        self.l4 = Tensor(random_uniform(1024, 784))

    def forward(self, x):
        for layer in [self.l1, self.l2, self.l3]:
            leakyrelu(x.dot(layer))
        return x.dot(self.l4).tanh()

class LinearDisc(nn):
    def __init__(self):
        self.l1 = Tensor(random_uniform(784, 1024))
        self.l2 = Tensor(random_uniform(1024, 512))
        self.l3 = Tensor(random_uniform(512, 256))
        self.l4 = Tensor(random_uniform(256, 2))

    def forward(self, x):
        for layer in [self.l1, self.l2, self.l3]:
            leakyrelu(x.dot(layer))
        return x.dot(self.l4).logsoftmax()
    
import gzip
def fetch(url):
    import requests, tempfile, os
    fp = os.path.join(tempfile.gettempdir(), url.encode()[-10:].hex())
    if os.path.isfile(fp) and os.stat(fp).st_size:
        with open(fp, 'rb') as f:
            return f.read()

    dat = requests.get(url).content
    with open(fp + '.tmp', 'wb') as f:
        f.write(dat)
    os.rename(fp+'.tmp', fp)
    return dat

def test_minst_gan():
    generator = LinearGen()
    discriminator = LinearDisc()

    parse = lambda dat: np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()
    x_train = parse(fetch(url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"))[0x10:].reshape((-1, 28*28)).astype(np.float32)

    # Hyperparameters
    epochs = 10
    batch_size = 512
    n_batches = int(len(x_train) / batch_size)
    output_folder = "outputs"

    ds_noise = np.random.randn(64,128).astype(np.float32)

    optim_g = Adam(generator.params, learning_rate=0.0002, beta1=0.5)
    optim_d = Adam(discriminator.params, learning_rate=0.0002, beta1=0.5)

    def batches_generator():
        batch_nr = 0
        while batch_nr < n_batches:
            idx = np.random.randint(0, x_train.shape[0], size=(batch_size))
            image_b = x_train[idx].reshape(-1, 28*28).astype(np.float32)/255.
            image_b = (image_b - 0.5)/0.5
            yield image_b
            batch_nr += 1

    def real_label(bs):
        y = np.zeros((bs,2), np.float32)
        y[range(bs), [1]*bs] = -2.0
        real_labels = Tensor(y)
        return real_labels

    def fake_label(bs):
        y = np.zeros((bs,2), np.float32)
        y[range(bs), [0]*bs] = -2.0
        fake_labels = Tensor(y)
        return fake_labels

    def train_discriminator(optim, data_real, data_fake):
        real_labels = real_label(batch_size)
        fake_labels = fake_label(batch_size)

        optim.zero_grad()

        output_real = discriminator.forward(data_real)
        loss_real = real_labels.mul(output_real).mean(axis=(0,1))

        output_fake = discriminator.forward(data_fake)
        loss_fake = fake_labels.mul(output_fake).mean(axis=(0,1))

        loss_real.backward()
        loss_fake.backward()
        optim.step()
        return loss_fake.val + loss_real.val

    def train_generator(optim, data_fake):
        real_labels = real_label(batch_size)
        optim.zero_grad()
        output = discriminator.forward(data_fake)
        loss = real_labels.mul(output).mean(axis=(0,1))
        loss.backward()
        optim.step()
        return loss.val

    for epoch in range(epochs):
        batches = tuple(batches_generator())
        for data_real in batches:
            data_real = Tensor(data_real)
            noise = Tensor(np.random.randn(batch_size, 128))
            data_fake = generator.forward(noise)
            data_fake = Tensor(data_fake.val)
            loss_d = train_discriminator(optim_d, data_real, data_fake).item()

            noise = Tensor(np.random.randn(batch_size, 128))
            data_fake = generator.forward(noise)
            loss_g = train_generator(optim_g, data_fake).item()

        # generate images after each epoch
        fake_images = generator.forward(Tensor(ds_noise)).val
        fake_images = (fake_images.reshape(-1, 1, 28, 28)+ 1) / 2
        fake_images = make_grid(torch.tensor(fake_images))
        save_image(fake_images, os.path.join(output_folder, f'image_{epoch}.jpg'))