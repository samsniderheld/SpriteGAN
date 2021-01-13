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
  parser.add_argument('--initial_gen_filters', type=int, default=512, help='The initial number of generator fitlers')
  parser.add_argument('--numLerps', type=int, default=10, help='How many lerps to do')
  parser.add_argument('--gen_kernel_size', type=int, default=3, help='The size of generator kernels')
  parser.add_argument('--disc_kernel_size', type=int, default=3, help='The size of the discriminator kernels')
  parser.add_argument('--noise_dim', type=int, default=128, help='The size of the latent vector')
  parser.add_argument('--img_dim', type=int, default=128, help='The dimension of the image')

  return parser.parse_args()

if os.path.exists("Results/GeneratedImages"):
  shutil.rmtree("Results/GeneratedImages")
  os.makedirs('Results/GeneratedImages')
else:
  os.makedirs('Results/GeneratedImages')

def f(x):
  return x


args = parse_args()

numLerps = args.numLerps
idx=0
noise_shape = (1,1,128)
noise_dim = 128

generator = load_generator_from_step(args)
generator.summary()


v1 = tf.random.truncated_normal([1,1,1,noise_dim])
v2 = tf.random.truncated_normal([1,1,1,noise_dim])

linX = list(np.linspace(0, 1, 50))

for i in tqdm(range(0,numLerps)):

  for x in linX:

    frame = None

    #use a linear interpolater 
    v = f(x) * v2 + f(1-x) * v1

    #get the output and reshape it 
    y = generator(v,training=False)
    y = denorm_img(y)
    y = np.array(y)

    cv2.imwrite('Results/GeneratedImages/image{}.png'.format('%04d'%idx), cv2.cvtColor(y[0], cv2.COLOR_RGB2BGR))
    
    idx+=1

  v1 = v2
  v2 = tf.random.truncated_normal([1,1,1,noise_dim])
