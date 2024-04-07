import tensorflow as tf
import numpy as np
import argparse

import sys
sys.path.append("../")
from img_generator import img_pair_gen1

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", dest = "number", default = 10,
                        help = "Number of examples")
    parser.add_argument("-w", dest = "write_to", default = None,
                        help = "Write to file")
    parser.add_argument("-r", dest = "read_from", default = None,
                        help = "Read from file")
    args = parser.parse_args()

    if args.write_to is not None:
        write(args.write_to, args.number)

    elif args.read_from is not None:
        read(args.read_from)
        for _, data_batch in zip(args.number, dataset.map(mapper)):
            print(data_batch)

def write(file, number):
    img_gen = img_pair_gen1(256, 256)

    with tf.io.TFRecordWriter(file) as file_writer:
        for _ in range(number):
            input, output = next(img_gen)

            input_serial = tf.io.serialize_tensor(input)
            output_serial = tf.io.serialize_tensor(output)

            record_bytes = tf.train.Example(features=tf.train.Features(feature={
                "i": tf.train.Feature(bytes_list =\
                    tf.train.BytesList(value = [input_serial.numpy()])),
                "o": tf.train.Feature(bytes_list =\
                    tf.train.BytesList(value = [output_serial.numpy()])),
            })).SerializeToString()
            file_writer.write(record_bytes)

def read(file):
    dataset = tf.data.TFRecordDataset(file)
    def mapper(from_ds):
        parsed_tensors = tf.io.parse_single_example(
            from_ds,
            {
                "i": tf.io.FixedLenFeature([], dtype = tf.string),
                "o": tf.io.FixedLenFeature([], dtype = tf.string)
            })
        return tf.io.parse_tensor(parsed_tensors["i"], tf.float32),\
               tf.io.parse_tensor(parsed_tensors["o"], tf.float32)

if __name__ == "__main__":
    main()
