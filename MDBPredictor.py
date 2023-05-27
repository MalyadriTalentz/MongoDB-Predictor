import tensorflow as tf
import pymongo

# Create a MongoDB Atlas client
client = pymongo.MongoClient(
    "mongodb://<username>:<password>@<cluster_url>/<database>"
)

# Get the data from MongoDB
data = client["<collection>"].find()

# Create a TensorFlow dataset
dataset = tf.data.Dataset.from_generator(
    lambda: data, output_types=(tf.float32, tf.float32)
)

# Create a TensorFlow model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="linear"),
])

# Train the model
model.compile(optimizer="sgd", loss="mse")
model.fit(dataset, epochs=10)

# Make a prediction
prediction = model.predict(data[0])

# Print the prediction
print(prediction)
