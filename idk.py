import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import os

def convert_pb_to_h5(pb_model_dir, h5_model_path):
    # Define the model architecture manually based on the original model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')  # Adjust the number of units as needed
    ])

    # Load the SavedModel using a `tf.Graph` approach
    model_path = os.path.join(pb_model_dir, 'saved_model.pb')
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Import the graph definition into a new graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

        # Create a TensorFlow session to extract weights
        with tf.compat.v1.Session(graph=graph) as sess:
            # Extract weights from the TensorFlow graph and assign them to the Keras model
            # Note: You may need to adapt this part to match your model's layers and weights structure

            # Assuming you know the layer names, you can use something like this
            for layer in model.layers:
                weights = sess.run(graph.get_tensor_by_name(f'{layer.name}/kernel:0'))
                biases = sess.run(graph.get_tensor_by_name(f'{layer.name}/bias:0'))
                layer.set_weights([weights, biases])

    # Save the Keras model in HDF5 format
    model.save(h5_model_path)
    print(f"Model saved as {h5_model_path}")

# Define file paths
pb_model_dir = 'saved_model'  # Directory containing the SavedModel files
h5_model_path = 'saved_model/model.h5'

# Convert the model
convert_pb_to_h5(pb_model_dir, h5_model_path)
