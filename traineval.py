from torch import optim
from config import ProcessConfig
from dataset import MelSpectrogramConfig, MelSpectrogram, LJSpeechDataset, LJSpeechCollator
from torch.utils.data import DataLoader, dataloader
from itertools import islice
from model import Disc, Generator
import torch.nn.functional as F
import torch
import scipy.io.wavfile
import os
import tqdm

def prepare_model_loader_losses(config: ProcessConfig):
    featurizer = MelSpectrogram(MelSpectrogramConfig()).to(config.device)
    dataset = LJSpeechDataset(config.datapath)
    collator = LJSpeechCollator()
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, collate_fn=collator)

    generator = Generator(config).to(config.device)
    optim_g = torch.optim.AdamW(generator.parameters(), lr=0.0002, betas=[0.8, 0.99])
    disc = Disc(config).to(config.device)
    optim_d = torch.optim.AdamW(disc.parameters(), lr=0.0002, betas=[0.8, 0.99])

    return dataloader, featurizer, generator, optim_g, disc, optim_d

real_label = 1.
fake_label = 0.

def train_batch(config: ProcessConfig, batch, featurizer, generator, optim_g, disc, optim_d):
    generator.train()

    mels = featurizer(batch.waveform.to(config.device))[:,:,:-1]
    wavs = batch.waveform.to(config.device)[:, :mels.shape[2]*256]
    wavs_estimated = generator(mels)
    mels_estimated = featurizer(wavs_estimated)[:,:,:mels.shape[2]]

    size = config.batch_size
    label = torch.full((size,), real_label, dtype=torch.float, device=config.device)


    optim_d.zero_grad()
    label.fill_(real_label)
    output = disc(wavs).view(-1)
    errD_real = F.binary_cross_entropy(output, label)
    errD_real.backward()
    label.fill_(fake_label)
    output = disc(wavs_estimated.detach()).view(-1)
    errD_fake = F.binary_cross_entropy(output, label)
    errD_fake.backward()
    errD = errD_real + errD_fake
    if errD.item() > 0.5:
        optim_d.step()


    optim_g.zero_grad()
    label.fill_(real_label)
    output = disc(wavs_estimated).view(-1)
    adv_loss = F.binary_cross_entropy(output, label)
    mel_loss = F.l1_loss(mels, mels_estimated) * 45
    loss_gen = adv_loss + mel_loss
    loss_gen.backward()
    optim_g.step()

    return loss_gen.item(), errD.item()

def train_checkpoint(config: ProcessConfig, steps=100):
    dataloader, featulizer, generator, optim_g, disc, optim_d = \
        prepare_model_loader_losses(config)
    batch = list(islice(dataloader, 1))[0]

    os.system('rm -r eval')
    os.system('mkdir eval')

    for i in range(steps):
        gen_loss, disc_loss = train_batch(
            config, batch, featulizer, generator, optim_g, disc, optim_d
        )
        if i % 50 == 0:
            print(i, gen_loss, disc_loss)
            eval(config, generator, str(i), featulizer)

def train(config: ProcessConfig, epochs=50):
    dataloader, featulizer, generator, optim_g, disc, optim_d = \
        prepare_model_loader_losses(config)

    os.system('rm -r eval')
    os.system('mkdir eval')

    i = 0
    for epoch in range(epochs):
        gloss = 0
        dloss = 0
        for batch in tqdm.tqdm(dataloader):
            loss_gen, loss_disc = train_batch(
                config, batch, featulizer, generator, optim_g, disc, optim_d
            )

            gloss += loss_gen
            dloss += loss_disc
            if (i + 1) % 100 == 0:
                eval(
                    config, generator, 
                    str(epoch).zfill(3)+'_'+str(i+1).zfill(7),
                    featulizer
                )
                print("Step:", i + 1, end=' ')
                print("Generator loss:", gloss/100, end=' ')
                print("Disc loss:", dloss/100, end=' ')
                print()
                gloss = 0
                dloss = 0
            i += 1

def eval(config: ProcessConfig, model, name: str, featulizer):
    model.eval()
    dataset = LJSpeechDataset(config.datapath)
    collator = LJSpeechCollator()
    dataloader = DataLoader(dataset, batch_size=3, collate_fn=collator)
    dummy_batch = list(islice(dataloader, 1))[0]

    with torch.no_grad():
        mels = featulizer(dummy_batch.waveform.to(config.device))
        wavs = model(mels)

    for i in range(wavs.shape[0]):
        wav = wavs[i].cpu().numpy()
        scipy.io.wavfile.write("eval/"+name+'_'+str(i) + '.wav', 22050, wav)
