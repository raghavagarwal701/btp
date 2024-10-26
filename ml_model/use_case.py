from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# Load the model
model = load_model('mc.h5')

# Preprocess the image
def preprocess_image(image_path, target_size):
    img = Image.open(image_path)
        
    img = img.resize(target_size)
    img_array = image.img_to_array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension (batch size)
    img_array = img_array / 255.0 # Normalize the image (same as training)
    img.save('preview.png')
    return img_array

# Predict the class
def predict_image(model, image_path, target_size=(130, 130)):
    img_array = preprocess_image(image_path, target_size)
    prediction = model.predict(img_array)
    return prediction

# Example usage
image_path = 'uinf_2.png'
result = predict_image(model, image_path)
print(result)
#add the result to result.txt file
f = open("result.txt", "w")
f.write(str(result) + "\n")

# # Interpret the result
predicted_class = np.argmax(result, axis=1)
class_labels = ['Parasitized', 'Uninfected']
predicted_label = class_labels[predicted_class[0]]
print(f"Predicted label: {predicted_label}")
#wriet in next line of file
f.write(f"Predicted label: {predicted_label}\n")

