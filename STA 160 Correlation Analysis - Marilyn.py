#!/usr/bin/env python
# coding: utf-8

# In[10]:


import cv2
import numpy as np
from scipy.stats import pearsonr

# List of image paths and corresponding sold prices
image_paths = [
    '/Users/kaylapham/Downloads/marilyn/blue.jpeg',
    '/Users/kaylapham/Downloads/marilyn/sage.png',
    '/Users/kaylapham/Downloads/marilyn/red.png',
    '/Users/kaylapham/Downloads/marilyn/orange.png',
    '/Users/kaylapham/Downloads/marilyn/turquoise.png']


sold_prices = [5000, 195000000, 4100000, 17300000, 80000000]


# ### Average Pixel Intensity Correlation

# In[11]:


# Define a function to extract image features (e.g., color histograms, texture features, etc.)
def extract_features(image):
    # Modify this function to extract relevant features from the image
    # For demonstration purposes, let's compute the average pixel intensity
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    average_intensity = np.mean(gray_image)
    return average_intensity

# Load and preprocess the images
images = []
for path in image_paths:
    image = cv2.imread(path)
    if image is None:
        print(f'Failed to load image: {path}')
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)

# Extract features from the images
image_features = [extract_features(image) for image in images]

# Calculate the correlation coefficient and p-value
correlation_coef, p_value = pearsonr(image_features, sold_prices)

# Print the correlation coefficient and p-value
print(f'Correlation coefficient: {correlation_coef}')
print(f'p-value: {p_value}')


# In[48]:


image_features


# ### Haralick Texture Features

# In[12]:


import mahotas
import numpy as np
from scipy.stats import spearmanr

def calculate_haralick_features(image):
    # Convert the image to grayscale and integer type
    gray_image = mahotas.colors.rgb2gray(image)
    gray_image = gray_image.astype(np.uint8)

    # Calculate Haralick features
    haralick_features = mahotas.features.haralick(gray_image)

    # Calculate the mean value of each Haralick feature
    mean_haralick_features = np.mean(haralick_features, axis=0)

    return mean_haralick_features

# Extract features from the images
haralick_features = [calculate_haralick_features(image) for image in images]


# Calculate the Spearman correlation coefficient and p-value
correlation_coef, p_value = spearmanr(haralick_features, sold_prices)

# Print the correlation coefficient and p-value
print(f'Spearman correlation coefficient: {correlation_coef}')
print(f'p-value: {p_value}')


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt
# Create a heatmap using seaborn
sns.set(style="white")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_coef, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, ax=ax)
ax.set_xticklabels([f"Var {i+1}" for i in range(correlation_coef.shape[0])])
ax.set_yticklabels([f"Var {i+1}" for i in range(correlation_coef.shape[0])])
plt.title("Haralick Features Correlation Matrix")
plt.show()


# In[14]:



# Define a function to extract color histogram features
def extract_color_histogram(image):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Calculate the color histogram
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # Return the color histogram feature
    return hist

# Load and preprocess the images
images = []
for path in image_paths:
    image = cv2.imread(path)
    if image is None:
        print(f'Failed to load image: {path}')
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)

# Extract color histogram features from the images
color_histograms = [extract_color_histogram(image) for image in images]


# Calculate the correlation coefficients
correlation_coef, p_value = spearmanr(color_histograms, sold_prices)

correlation_coef = np.nan_to_num(correlation_coef)
# Print the correlation coefficient and p-value
print(f'Spearman correlation coefficient: {correlation_coef}')
print(f'p-value: {p_value}')


# In[8]:


correlation_coef


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
# Create a heatmap using seaborn
sns.set(style="white")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_coef, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, ax=ax)

plt.title("Color Histogram Correlation Matrix")
plt.show()


# In[9]:


sns.heatmap(correlation_coef)

