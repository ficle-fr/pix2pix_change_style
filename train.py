import tensorflow as tf
import numpy as np

import time
import os

from matplotlib import pyplot as plt

from tqdm import tqdm

from generator import Generator, generator_loss
from descriminator import Discriminator, discriminator_loss

from img_generator import img_pair_gen1


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

@tf.function
def train_step(generator, discriminator, generator_optimizer, discriminator_optimizer,
               input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training = True)

        disc_real_output = discriminator([input_image, target], training = True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    #with summary_writer.as_default():
    #    tf.summary.scalar('gen_total_loss', gen_total_loss, step = step//1000)
    #    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step = step//1000)
    #    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step = step//1000)
    #    tf.summary.scalar('disc_loss', disc_loss, step = step//1000)

def fit(generator, discriminaror, generator_optimizer, discriminator_optimizer, 
        train_ds, test_ds, epochs, steps_in_epoch, after_each_epoch_callback):
    example_input, example_target = next(iter(test_ds.take(1)))
    generate_images(generator, example_input, example_target)
    start = time.time()

    for _ in range(epochs):
        start = time.time()
        for input_image, target in tqdm(train_ds.take(steps_in_epoch), total = steps_in_epoch):
            train_step(generator, discriminaror, generator_optimizer, discriminator_optimizer, input_image, target)
        print(f'\nTime taken for 1 epoch: {time.time()-start:.2f} sec\n')
        generate_images(generator, example_input, example_target)
        after_each_epoch_callback()

def main():
    generator = Generator([256, 256, 3], 3)
    discriminator = Discriminator([256, 256, 3])

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    #generator_optimizer = tf.keras.optimizers.Adam(5e-5, beta_1=0.5)
    #discriminator_optimizer = tf.keras.optimizers.Adam(5e-5, beta_1=0.5)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                     discriminator_optimizer = discriminator_optimizer,
                                     generator = generator,
                                     discriminator = discriminator)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    def checkpoint_callback():
        checkpoint.save(file_prefix = checkpoint_prefix)


    BATCH_SIZE = 5
    dataset = tf.data.Dataset.from_generator(
        lambda: img_pair_gen1(256, 256),
        output_signature = (tf.TensorSpec(shape = (256, 256, 3), dtype = tf.float32),
                            tf.TensorSpec(shape = (256, 256, 3), dtype = tf.float32)))
    dataset = dataset.batch(BATCH_SIZE)

    #fit(dataset, dataset, 40, 100)
    fit(generator, discriminator, generator_optimizer, discriminator_optimizer,
        dataset, dataset, 40, 50, checkpoint_callback)

if __name__ == "__main__":
    main()
