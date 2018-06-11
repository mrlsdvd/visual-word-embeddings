import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
from build_vocab import Vocabulary
from model import EncoderCNN, Autoencoder
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from tensorboardX import SummaryWriter
import csv
import sys
# Add path to config
sys.path.append('../../')
import config as conf


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    autoencoder = Autoencoder(args.embed_size, args.embeddings_path, args.hidden_size, len(vocab), args.num_layers).to(device)
    print(len(vocab))

    # optimizer
    params = list(filter(lambda p: p.requires_grad, list(autoencoder.parameters())[1:] + list(encoder.linear.parameters())))
    # print(params)
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Define summary writer
    writer = SummaryWriter()

    # Loss tracker
    best_loss = float('inf')

    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            # print(captions)
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward, backward and optimize
            features = encoder(images)
            L_ling, L_vis = autoencoder(features, captions, lengths)
            loss = 0.2*L_ling + 0.8*L_vis # Want visual loss to have bigger impact
            autoencoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Save the model checkpoints when loss improves
            if loss.item() < best_loss:
                best_loss = loss
                print("Saving checkpoints")
                torch.save(autoencoder.state_dict(), os.path.join(
                    args.model_path, 'autoencoder-frozen-best.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder-frozen-best.ckpt'.format(epoch+1, i+1)))

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
                # Log train loss on tensorboard
                writer.add_scalar('frozen-loss/L_ling', L_ling.item(), epoch*total_step + i)
                writer.add_scalar('frozen-loss/L_vis', L_vis.item(), epoch*total_step + i)
                writer.add_scalar('frozen-loss/combined', loss.item(), epoch*total_step + i)

            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(autoencoder.state_dict(), os.path.join(
                    args.model_path, 'autoencoder-frozen-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder-frozen-{}-{}.ckpt'.format(epoch+1, i+1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--embeddings_path', type=str, default=os.path.join(conf.models_path, 'glove', 'glove.6B', 'glove.6B.200d.txt'),
                        help='path for pretrained embeddings')
    parser.add_argument('--image_dir', type=str, default='data/resized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=20, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int , default=200, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=256, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
