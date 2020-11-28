import argparse
import os
import tensorflow as tf
import tensorflow.lite as lite

parser = argparse.ArgumentParser(description='Convert a Tensorflow Model to TFLite.')
parser.add_argument('--triplet_model', type=str, help='Path to a savedmodel of triplet model',
                    default='saved_models/triplet/efficientnet_v3.0')
parser.add_argument('--output', type=str, help='output tag', default='v1.0')
args = parser.parse_args()

model_path = args.triplet_model
output = args.output

out_path = os.path.join(os.path.dirname(__file__), 'output', output)
if not os.path.exists(out_path):
    os.makedirs(out_path)

print("Loading model from {}".format(model_path))
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
print("Converting...")
tflite_model = converter.convert()
print("Saving to {}".format(out_path))
open(os.path.join(out_path, "model.tflite"), "wb").write(tflite_model)