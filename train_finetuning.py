import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

train_path = r"E:\Mobilenetv2\Dataset\train"
validation_path = r"E:\Mobilenetv2\Dataset\validate"
test_path = r"E:\Mobilenetv2\Dataset\test"

# Data generators
train_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest').flow_from_directory(
    train_path, target_size=(224, 224), color_mode='rgb',
    batch_size=32, class_mode='categorical', shuffle=True)

validation_generator = ImageDataGenerator(
    rescale=1./255).flow_from_directory(
    validation_path, target_size=(224, 224), color_mode='rgb',
    batch_size=32, class_mode='categorical', shuffle=True)

test_generator = ImageDataGenerator(
    rescale=1./255).flow_from_directory(
    test_path, target_size=(224, 224), color_mode='rgb',
    batch_size=32, class_mode='categorical', shuffle=False)

# Build the model
base_model = MobileNetV2(weights='imagenet', include_top=False )
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
predict_layer = Dense(15, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predict_layer)



print(model.summary())
# Initially freeze the entire base model
for layer in base_model.layers:
    layer.trainable = False
# Compile the model
#optimizer = Adam(learning_rate=0.0001)
optimizer = RMSprop(0.001)
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)
numOfEpochs = 10
# Train the model with frozen base model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=numOfEpochs
)

# Unfreeze the last few layers of the base model for fine-tuning
for layer in base_model.layers[100:]:
    layer.trainable = True
model_checkpoint = ModelCheckpoint(
    'E:\Mobilenetv2\Train_MobileNetV2\checkpoint\mushroom_model-{epoch:03d}.h5',
    save_best_only=True,
    verbose=1,
    monitor='val_loss',
    mode='auto'
)
early_stopping = EarlyStopping(
    patience=10,
    restore_best_weights=True,
    monitor='val_loss'
)
f1_score = F1Score(num_classes=15, average='weighted')

callbacks = [model_checkpoint, early_stopping]


# Recompile the model with a lower learning rate
#optimizer = Adam(learning_rate=0.00001)
numOfEpoch = 50
optimizer = SGD(0.001)
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy', F1Score(num_classes=15, average='weighted')]
)

# Fine-tune the model
history_finetune = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=numOfEpoch,
    callbacks=callbacks
)

model.save("E:\Mobilenetv2\savemodel\mushroomv2_finetuning_50.h5")

# Evaluate the model on the test dataset
test_loss, test_accuracy, test_f1_score = model.evaluate(test_generator)

print(f'Test Accuracy: {test_accuracy}')
print(f'Test F1-Score: {test_f1_score}')

# Get the true labels and predictions for the test set
test_generator.reset()
y_true = []
y_pred = []

for i in range(len(test_generator)):
    X, y = test_generator[i]
    y_true.extend(np.argmax(y, axis=1))
    y_pred.extend(np.argmax(model.predict(X), axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=list(test_generator.class_indices.keys()))

# Plot the confusion matrix with vertical x-axis labels
plt.figure(figsize=(10, 10))
cmd.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=90)  # Rotate x-axis labels
plt.title('Confusion Matrix')
plt.show()

# Plot training and validation metrics
train_acc = history.history['accuracy'] + history_finetune.history['accuracy']
train_loss = history.history['loss'] + history_finetune.history['loss']
val_acc = history.history['val_accuracy'] + history_finetune.history['val_accuracy']
val_loss = history.history['val_loss'] + history_finetune.history['val_loss']

# Find the best epochs for validation loss and accuracy
index_loss = np.argmin(val_loss)
index_acc = np.argmax(val_acc)

val_lowest = val_loss[index_loss]
val_highest = val_acc[index_acc]

# Generate a list of epoch numbers
Epochs = [i + 1 for i in range(len(train_acc))]

# Labels for the best epochs
loss_label = f'Best Epoch = {str(index_loss + 1)} (Loss: {val_lowest:.4f})'
acc_label = f'Best Epoch = {str(index_acc + 1)} (Accuracy: {val_highest:.4f})'

# Plotting the results
plt.figure(figsize=(20, 8))
plt.style.use('fivethirtyeight')

# Plot for loss
plt.subplot(1, 2, 1)
plt.plot(Epochs, train_loss, 'r', label='Training Loss')
plt.plot(Epochs, val_loss, 'g', label='Validation Loss')
plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot for accuracy
plt.subplot(1, 2, 2)
plt.plot(Epochs, train_acc, 'r', label='Training Accuracy')
plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
plt.scatter(index_acc + 1, val_highest, s=150, c='blue', label=acc_label)
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
