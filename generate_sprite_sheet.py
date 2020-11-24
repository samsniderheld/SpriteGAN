import argparse
import os
import tensorflow as tf
from Model.load_model import load_generator_from_step
from Utils.reporting import denorm_img
import cv2
import shutil
import numpy as np
from tqdm import tqdm

def parse_args():
  desc = "create a lerp video"
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('--step', type=int, default=10000, help='The step to load the  model from ')
  parser.add_argument('--gen_kernel_size', type=int, default=3, help='The size of generator kernels')
  parser.add_argument('--disc_kernel_size', type=int, default=3, help='The size of the discriminator kernels')
  parser.add_argument('--noise_dim', type=int, default=128, help='The size of the latent vector')
  parser.add_argument('--img_dim', type=int, default=128, help='The dimension of the image')
  parser.add_argument('--sprite_sheet_dim', type=int, default=10, help='The Sprite Sheet Dim')

  return parser.parse_args()

if os.path.exists("Results/SpriteSheets"):
  shutil.rmtree("Results/SpriteSheets")
  os.makedirs('Results/SpriteSheets')
else:
  os.makedirs('Results/SpriteSheets')

def f(x):
  return x


args = parse_args()

idx=0
noise_dim = args.noise_dim
noise_shape = (1,1,args.noise_dim)
sprite_sheet_dim = args.sprite_sheet_dim
imgs = []

generator = load_generator_from_step(args)
generator.summary()


v1 = tf.random.truncated_normal([1,1,1,noise_dim])
v2 = tf.random.truncated_normal([1,1,1,noise_dim])

linX = list(np.linspace(0, 1, sprite_sheet_dim))

startV = v1

for i in tqdm(range(0,sprite_sheet_dim)):

  for x in linX:

    frame = None

    #use a linear interpolater 
    v = f(x) * v2 + f(1-x) * v1

    #get the output and reshape it 
    y = generator(v,training=False)
    y = denorm_img(y)
    y = np.array(y[0])

    imgs.append(y)

    idx+=1

  if i  == sprite_sheet_dim - 2 :
    v1 = v2
    v2 = startV
  else:
    v1 = v2
    v2 = tf.random.truncated_normal([1,1,1,noise_dim])

v_stacks = []

for i in range(0,pow(sprite_sheet_dim,2),sprite_sheet_dim):
  v_stacks.append(cv2.hconcat(imgs[i:i+sprite_sheet_dim]))

full_img = cv2.vconcat(v_stacks)

cv2.imwrite('Results/SpriteSheets/SpriteSheet.png', cv2.cvtColor(full_img, cv2.COLOR_RGB2BGR))
