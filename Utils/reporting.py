from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2

def denorm_img(img):
    img = (img + 1) * 127.5
    return img

def generate_and_save_images(model, step, test_input):
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))
  gs1 = gridspec.GridSpec(3, 3)
  gs1.update(wspace=0, hspace=0)

  for i in range(predictions.shape[0]):

      ax1 = plt.subplot(gs1[i])
      ax1.set_aspect('equal')
      generated_image= denorm_img(predictions[i, :, :, :]) / 255.
      fig = plt.imshow(generated_image)
      plt.axis('off')
      fig.axes.get_xaxis().set_visible(False)
      fig.axes.get_yaxis().set_visible(False)

  plt.tight_layout()
  plt.savefig('Results/Images/Distribution/distribution_image_at_step_{:06d}.png'.format(step))
  plt.show()

  single_image_1 = denorm_img(predictions[1, :, :, :]) / 255.
  plt.imshow(single_image_1)
  plt.show()

  saveImg = cv2.resize(np.asarray(single_image_1)*255, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
  cv2.imwrite('Results/Images/SingleImage/single_image_1_at_step_{:06d}.png'.format(step), cv2.cvtColor(saveImg, cv2.COLOR_RGB2BGR))


  plt.close('all')


def test_dataset(imgs):

  fig = plt.figure(figsize=(4,4))
  gs1 = gridspec.GridSpec(3, 3)
  gs1.update(wspace=0, hspace=0)

  for i in range(9):

      ax1 = plt.subplot(gs1[i])
      ax1.set_aspect('equal')
      image= denorm_img(imgs[i]) / 255.
      fig = plt.imshow(image)
      plt.axis('off')
      fig.axes.get_xaxis().set_visible(False)
      fig.axes.get_yaxis().set_visible(False)

  plt.tight_layout()
  plt.savefig('Results/Images/dataset_representation.png')
  plt.show()


def plot_loss(all_disc_loss,all_gen_loss):

  plt.figure(figsize=(10,5))
  plt.plot(np.arange(len(all_disc_loss)),all_disc_loss,label='D')
  plt.plot(np.arange(len(all_gen_loss)),all_gen_loss,label='G')
  plt.ylim(-50,50)
  plt.legend()
  plt.title('All Time Loss')
  plt.savefig('Results/Images/Loss/all_losses')
  plt.show()
  
  plt.close('all')



