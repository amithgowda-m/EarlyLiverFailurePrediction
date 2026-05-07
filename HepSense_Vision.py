import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"TensorFlow Version: {tf.__version__}")
print("Setting up the HepSense Vision Pipeline...")

# 1. Point to the folders
# UPDATE THIS PATH if your F0-F4 folders are nested deeper (e.g., "Dataset/Dataset")
data_dir = "Dataset" 

# 2. The Data Loader
print("\nLoading Training Data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=42,
  image_size=(224, 224),
  batch_size=32
)

print("Loading Validation Data...")
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=42,
  image_size=(224, 224),
  batch_size=32
)

# 3. Data Augmentation (To fix the class imbalance)
data_augmentation = keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.1),
])

# 4. Load the Pre-trained Brain (MobileNetV2)
print("\nDownloading MobileNetV2 Base Brain...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False, 
    weights='imagenet' 
)
base_model.trainable = False # Freeze the core knowledge

# 5. Build the Final HepSense Vision Model
print("Attaching Custom Classification Layers...")
inputs = keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(5, activation='softmax')(x) # 5 stages: F0 to F4

model = keras.Model(inputs, outputs)

# 6. Compile the AI
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# 7. Initiate the Training Sequence
print("\n=======================================================")
print("INITIATING SPRINT 2: TRAINING VISION PIPELINE")
print("=======================================================")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10 
)

# 8. Save the Brain!
model.save('hepsense_vision_v1.keras')
print("\nTraining Complete! Model saved locally as 'hepsense_vision_v1.keras'")