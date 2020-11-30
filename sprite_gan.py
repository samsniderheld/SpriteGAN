import argparse
from Training.training import train
from Training.training_distributed import *
import os
import shutil


def parse_args():
	desc = "A GAN designed to generate sprites"

	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--data_dir', type=str, default="Data/SaganRaw/", help='The directory that holds the image data')
	parser.add_argument('--num_training_steps', type=int, default=10000, help='The number of batches to train on')
	parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
	parser.add_argument('--gen_kernel_size', type=int, default=3, help='The size of generator kernels')
	parser.add_argument('--disc_kernel_size', type=int, default=3, help='The size of the discriminator kernels')
	parser.add_argument('--noise_dim', type=int, default=128, help='The size of the latent vector')
	parser.add_argument('--img_dim', type=int, default=128, help='The dimension of the image')
	parser.add_argument('--d_lr', type=float, default=.0004, help='The initial discriminator lr')
	parser.add_argument('--g_lr', type=float, default=.0001, help='The initial generator lr')
	parser.add_argument('--no_scheduler', action='store_true')
	parser.add_argument('--print_freq', type=int, default=100, help='How often is the status printed')
	parser.add_argument('--save_freq', type=int, default=1000, help='How often is the model saved')
	parser.add_argument('--distributed', action='store_true')

	return parser.parse_args()


def main():

	if os.path.exists("Results/"):
		shutil.rmtree("Results/")
		os.makedirs("Results/")
		os.makedirs("Results/Images")
		os.makedirs("Results/Images/Distribution")
		os.makedirs("Results/Images/Loss")
		os.makedirs("Results/Images/SingleImage")
		os.makedirs("Results/GeneratedImages")
		os.makedirs("Results/LerpedVideos")
	else:
		os.makedirs("Results/")
		os.makedirs("Results/Images")
		os.makedirs("Results/Images/Distribution")
		os.makedirs("Results/Images/Loss")
		os.makedirs("Results/Images/SingleImage")
		os.makedirs("Results/GeneratedImages")
		os.makedirs("Results/LerpedVideos")

	args = parse_args()

	if(args.distributed):

		trainer = DistributedTrainer(args)
		trainer.train(args)

	else:
		train(args)

	print("done training")


if __name__ == '__main__':
	main()
