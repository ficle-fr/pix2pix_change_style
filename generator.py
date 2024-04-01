import tensorflow as tf

from common import downsample, upsample

def Generator(shape, output_channels):
    inputs = tf.keras.layers.Input(shape = shape)

    down_stack = [
        downsample(64, 4, apply_batchnorm = False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]

    up_stack = [
        upsample(512, 4, apply_dropout = True),
        upsample(512, 4, apply_dropout = True),
        upsample(512, 4, apply_dropout = True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4), 
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(output_channels, 4,
                                           strides = 2,
                                           padding = 'same',
                                           kernel_initializer = initializer,
                                           activation= 'tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs = inputs, outputs = x)

LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output),
                           disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

def main():
    input = tf.keras.layers.Input(shape = [256, 256, 3])
    conv_layer = tf.keras.layers.Conv2D(64, 3, strides = 2, padding = "same")
    convt_layer = tf.keras.layers.Conv2DTranspose(64, 3, strides = 1, padding = "same")
    output = conv_layer(input)
    output = convt_layer(output)
    model = tf.keras.models.Model(inputs = input, outputs = output)
    model.summary()


    #generator = Generator([256, 256, 3], 3)
    #generator.summary()

if __name__ == "__main__":
    main()
