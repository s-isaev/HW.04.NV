from librosa.filters import mel
from numpy import dtype
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
import wandb

def prepare_model_loader_losses(config: ProcessConfig, ignore_dataset=False):
    featurizer = MelSpectrogram(MelSpectrogramConfig()).to(config.device)

    dataloader = None
    if not ignore_dataset:
        dataset = LJSpeechDataset(config.datapath)
        collator = LJSpeechCollator()
        dataloader = DataLoader(
            dataset, batch_size=config.batch_size, collate_fn=collator
        )

    generator = Generator(config).to(config.device)
    optim_g = torch.optim.AdamW(generator.parameters(), lr=0.0002, betas=[0.8, 0.99])
    disc = Disc(config).to(config.device)
    optim_d = torch.optim.AdamW(disc.parameters(), lr=0.0002, betas=[0.8, 0.99])

    return dataloader, featurizer, generator, optim_g, disc, optim_d

real_label = 1.
fake_label = 0.

def train_batch(config: ProcessConfig, batch, featurizer, generator, optim_g, disc, optim_d):
    generator.train()
    disc.train()

    mels = featurizer(batch.waveform.to(config.device))[:,:,:-1]
    wavs = batch.waveform.to(config.device)[:, :mels.shape[2]*256]
    wavs_estimated = generator(mels)
    mels_estimated = featurizer(wavs_estimated)[:,:,:mels.shape[2]]

    size = wavs.shape[0]
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

    wandb.init(project=config.project, entity=config.entity)

    os.system('rm -r eval')
    os.system('mkdir eval')
    os.system('rm -r infer_res')
    os.system('mkdir infer_res')

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
                infer(
                    config, generator,
                    str(epoch).zfill(3)+'_'+str(i+1).zfill(7),
                    featulizer, step=i
                )
                eval(
                    config, generator, 
                    str(epoch).zfill(3)+'_'+str(i+1).zfill(7),
                    featulizer, step=i
                )
                # print("Step:", i + 1, end=' ')
                # print("Generator loss:", gloss/100, end=' ')
                # print("Disc loss:", dloss/100, end=' ')
                # print()
                wandb.log({"generator_loss": gloss/100}, step=int(i+1))
                wandb.log({"discriminator_loss": dloss/100}, step=int(i+1))
                gloss = 0
                dloss = 0

            if i % 500 == 0:
                torch.save(generator.state_dict(), config.model_path+'/'+str(i).zfill(7)+".pth")
            i += 1

def eval(config: ProcessConfig, model, name: str, featulizer, step=None):
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
        if step is not None:
            wandb.log(
                {"lg_audio_"+str(i): wandb.Audio(wav, sample_rate=22050)},
                step=step+1
            )
        scipy.io.wavfile.write("eval/"+name+'_'+str(i) + '.wav', 22050, wav)

def infer(config: ProcessConfig, model, name: str, featulizer, step=None):
    model.eval()

    for filename in os.listdir(config.infer_path):
        fname = config.infer_path + '/' + filename
        audio = scipy.io.wavfile.read(fname)[1]

        with torch.no_grad():
            wav = torch.from_numpy(audio).to(config.device).unsqueeze(0)
            if wav.dtype == torch.int16:
                wav = wav/wav.abs().max()
            mel = featulizer(wav)
            wav_synt = model(mel)

        wav_synt = wav_synt.cpu().numpy()[0]
        if step is not None:
            wandb.log(
                {filename: wandb.Audio(wav_synt, sample_rate=22050)},
                step=step+1
            )
        scipy.io.wavfile.write("infer_res/"+name+'_'+filename, 22050, wav_synt)