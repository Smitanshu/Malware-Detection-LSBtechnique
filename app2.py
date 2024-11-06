import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs

import cv2
import numpy as np
import gradio as gr
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cryptography.fernet import Fernet, InvalidToken
import base64
import hashlib

# Custom exception for steganography errors
class SteganographyException(Exception):
    pass

# Class for Least Significant Bit (LSB) Steganography
class LSBSteg():
    def __init__(self, im):
        self.image = im
        self.height, self.width, self.nbchannels = im.shape
        self.size = self.width * self.height

        # Masks for setting and clearing the least significant bit
        self.maskONEValues = [1, 2, 4, 8, 16, 32, 64, 128]
        self.maskONE = self.maskONEValues.pop(0)

        self.maskZEROValues = [254, 253, 251, 247, 239, 223, 191, 127]
        self.maskZERO = self.maskZEROValues.pop(0)

        self.curwidth = 0
        self.curheight = 0
        self.curchan = 0

    # Method to put binary value into the image
    def put_binary_value(self, bits):
        for c in bits:
            val = list(self.image[self.curheight, self.curwidth])
            if int(c) == 1:
                val[self.curchan] = int(val[self.curchan]) | self.maskONE
            else:
                val[self.curchan] = int(val[self.curchan]) & self.maskZERO

            self.image[self.curheight, self.curwidth] = tuple(val)
            self.next_slot()

    # Method to move to the next slot in the image
    def next_slot(self):
        if self.curchan == self.nbchannels - 1:
            self.curchan = 0
            if self.curwidth == self.width - 1:
                self.curwidth = 0
                if self.curheight == self.height - 1:
                    self.curheight = 0
                    if self.maskONE == 128:
                        raise SteganographyException("No available slot remaining (image filled)")
                    else:
                        self.maskONE = self.maskONEValues.pop(0)
                        self.maskZERO = self.maskZEROValues.pop(0)
                else:
                    self.curheight += 1
            else:
                self.curwidth += 1
        else:
            self.curchan += 1

    # Method to read a single bit from the image
    def read_bit(self):
        val = self.image[self.curheight, self.curwidth][self.curchan]
        val = int(val) & self.maskONE
        self.next_slot()
        if val > 0:
            return "1"
        else:
            return "0"

    # Method to read a byte (8 bits) from the image
    def read_byte(self):
        return self.read_bits(8)

    # Method to read a specified number of bits from the image
    def read_bits(self, nb):
        bits = ""
        for i in range(nb):
            bits += self.read_bit()
        return bits

    # Method to convert a value to its binary representation
    def byteValue(self, val):
        return self.binary_value(val, 8)

    # Method to convert a value to a binary string of a specified size
    def binary_value(self, val, bitsize):
        binval = bin(val)[2:]
        if len(binval) > bitsize:
            raise SteganographyException("binary value larger than the expected size")
        while len(binval) < bitsize:
            binval = "0" + binval
        return binval

    # Method to decode text from the image
    def decode_text(self):
        ls = self.read_bits(32)
        l = int(ls, 2)
        i = 0
        unhideTxt = bytearray()
        while i < l:
            tmp = self.read_byte()
            i += 1
            unhideTxt.append(int(tmp, 2))
        return unhideTxt.decode('utf-8')

# Load and preprocess images
def load_data(dataset_path):
    images = []
    labels = []
    for folder in os.listdir(dataset_path):
        label = folder
        folder_path = os.path.join(dataset_path, folder)
        
        if not os.path.isdir(folder_path):  # Ensure it's a directory
            continue
        
        for img in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img)
            image = cv2.imread(img_path)

            # Check if image is read correctly
            if image is None:
                print(f"Warning: Unable to read image {img_path}. Skipping...")
                continue

            image = cv2.resize(image, (128, 128))
            images.append(image)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Load the dataset
DATASET_PATH = r'D:\EDI-Project\Implementation\LSBSteganography\Datasets\malimg_dataset\train'
images, labels = load_data(DATASET_PATH)

# Normalize the images
images = images / 255.0

# Encode the labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=20)
datagen.fit(X_train)

# Build the CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')  # Number of classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and train the model
model = create_model()
model.fit(datagen.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test), epochs=2)

# Save the model
model.save('malware_classifier.h5')

# Function to classify potential threats based on keywords
def classify_threat(message):
    threat_keywords = ['malware', 'virus', 'trojan', 'worm', 'link']
    for keyword in threat_keywords:
        if keyword in message.lower():
            return f"Potential Threat Detected: {keyword.capitalize()}"
    return "No threats detected."

# Function to decode secret text from an encoded image
def decode_text_image(encoded_image):
    try:
        in_img = cv2.imread(encoded_image.name)
        steg = LSBSteg(in_img)
        hidden_text = steg.decode_text()
        return hidden_text
    except Exception as e:
        return "Error decoding the image."

# Function to classify an uploaded malware image
def classify_image(image_path):
  #  image_path = r"D:\EDI-Project\Implementation\LSBSteganography\encoded_image.png"
    
    image = cv2.imread(image_path)
    
    image = cv2.resize(image, (128, 128)) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    predictions = model.predict(image)
    class_index = np.argmax(predictions)
    return label_encoder.inverse_transform([class_index])[0]


# Combined function to classify and extract hidden messages
def analyze_image(image):
    hidden_message = decode_text_image(image)
    malware_type = classify_image(image)
    threat_warning = classify_threat(hidden_message)
    return f"Hidden Message: {hidden_message}\nDetected Malware Type: {malware_type}\nThreat Warning: {threat_warning}"

# Gradio interface for analysis
gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(type="filepath", label="Upload Malware Image"),
    outputs="text",
    title="Malware Detection and Message Extraction",
    description="Upload an image that may contain malware or hidden messages. The model will classify the type of malware, and any hidden messages will be extracted."
).launch()
