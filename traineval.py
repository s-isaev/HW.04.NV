from torch import optim
from config import ProcessConfig
from dataset import MelSpectrogramConfig, MelSpectrogram, LJSpeechDataset, LJSpeechCollator
from torch.utils.data import DataLoader, dataloader
from itertools import islice
from model import Generator
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

    return dataloader, featurizer, generator, optim_g

def train_batch(config: ProcessConfig, batch, featurizer, generator, optim_g):
    generator.train()

    mels = featurizer(batch.waveform.to(config.device))[:,:,:-1]
    wavs = batch.waveform.to(config.device)[:, :mels.shape[2]*256]

    wavs_estimated = generator(mels)
    mels_estimated = featurizer(wavs_estimated)[:,:,:mels.shape[2]]

    
    # Optimize Generator
    optim_g.zero_grad()
    loss_mel = F.l1_loss(mels, mels_estimated) * 45
    loss_gen = loss_mel
    loss_gen.backward()
    optim_g.step()

    return loss_gen.item()

def train_checkpoint(config: ProcessConfig, steps=100):
    dataloader, featulizer, generator, optim_g = \
        prepare_model_loader_losses(config)
    batch = list(islice(dataloader, 1))[0]

    os.system('rm -r eval')
    os.system('mkdir eval')

    for i in range(steps):
        gen_loss = train_batch(config, batch, featulizer, generator, optim_g)
        if i % 50 == 0:
            print(i, gen_loss)
            eval(config, generator, str(i), featulizer)

def train(config: ProcessConfig, epochs=50):
    dataloader, featulizer, generator, optim_g = \
        prepare_model_loader_losses(config)

    os.system('rm -r eval')
    os.system('mkdir eval')

    for epoch in range(epochs):
        i = 0
        gloss = 0
        for batch in tqdm.tqdm(dataloader):
            loss_gen = train_batch(
                config, batch, featulizer, generator, optim_g
            )

            gloss += loss_gen
            if (i + 1) % 100 == 0:
                eval(config, generator, str(batch).zfill(3)+'_'+str(i).zfill(5))
                print("Step:", i + 1, end=' ')
                print("Generator loss:", gloss/100, end=' ')
                gloss = 0
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
