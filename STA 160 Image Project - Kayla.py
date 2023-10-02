#!/usr/bin/env python
# coding: utf-8

# In[6]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
turquoise = cv2.imread('/Users/kaylapham/Downloads/turquoise.png')
sage = cv2.imread('/Users/kaylapham/Downloads/sage.png')
red = cv2.imread('/Users/kaylapham/Downloads/red.png')
orange = cv2.imread('/Users/kaylapham/Downloads/orange.png')
blue = cv2.imread('/Users/kaylapham/Downloads/blue.jpeg')


# In[7]:


# Check if the image was successfully loaded
if turquoise is None:
    print('Failed to load image')
    exit()

# Get image dimensions
height, width, channels = turquoise.shape
print('Image dimensions: {} x {}'.format(width, height))
print('Number of channels: {}'.format(channels))

# Convert the image to grayscale
gray_image = cv2.cvtColor(turquoise, cv2.COLOR_BGR2GRAY)

# Calculate the average pixel intensity
average_intensity = np.mean(gray_image)
print('Average pixel intensity: {}'.format(average_intensity))

# Calculate the maximum pixel intensity
max_intensity = np.max(gray_image)
print('Maximum pixel intensity: {}'.format(max_intensity))

# Calculate the minimum pixel intensity
min_intensity = np.min(gray_image)
print('Minimum pixel intensity: {}'.format(min_intensity))


# In[8]:


# Check if the image was successfully loaded
if sage is None:
    print('Failed to load image')
    exit()

# Get image dimensions
height, width, channels = sage.shape
print('Image dimensions: {} x {}'.format(width, height))
print('Number of channels: {}'.format(channels))

# Convert the image to grayscale
gray_image = cv2.cvtColor(sage, cv2.COLOR_BGR2GRAY)

# Calculate the average pixel intensity
average_intensity = np.mean(gray_image)
print('Average pixel intensity: {}'.format(average_intensity))

# Calculate the maximum pixel intensity
max_intensity = np.max(gray_image)
print('Maximum pixel intensity: {}'.format(max_intensity))

# Calculate the minimum pixel intensity
min_intensity = np.min(gray_image)
print('Minimum pixel intensity: {}'.format(min_intensity))


# In[9]:


# Check if the image was successfully loaded
if red is None:
    print('Failed to load image')
    exit()

# Get image dimensions
height, width, channels = red.shape
print('Image dimensions: {} x {}'.format(width, height))
print('Number of channels: {}'.format(channels))

# Convert the image to grayscale
gray_image = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)

# Calculate the average pixel intensity
average_intensity = np.mean(gray_image)
print('Average pixel intensity: {}'.format(average_intensity))

# Calculate the maximum pixel intensity
max_intensity = np.max(gray_image)
print('Maximum pixel intensity: {}'.format(max_intensity))

# Calculate the minimum pixel intensity
min_intensity = np.min(gray_image)
print('Minimum pixel intensity: {}'.format(min_intensity))


# In[10]:


# Check if the image was successfully loaded
if orange is None:
    print('Failed to load image')
    exit()

# Get image dimensions
height, width, channels = orange.shape
print('Image dimensions: {} x {}'.format(width, height))
print('Number of channels: {}'.format(channels))

# Convert the image to grayscale
gray_image = cv2.cvtColor(orange, cv2.COLOR_BGR2GRAY)

# Calculate the average pixel intensity
average_intensity = np.mean(gray_image)
print('Average pixel intensity: {}'.format(average_intensity))

# Calculate the maximum pixel intensity
max_intensity = np.max(gray_image)
print('Maximum pixel intensity: {}'.format(max_intensity))

# Calculate the minimum pixel intensity
min_intensity = np.min(gray_image)
print('Minimum pixel intensity: {}'.format(min_intensity))


# In[11]:


# Check if the image was successfully loaded
if blue is None:
    print('Failed to load image')
    exit()

# Get image dimensions
height, width, channels = blue.shape
print('Image dimensions: {} x {}'.format(width, height))
print('Number of channels: {}'.format(channels))

# Convert the image to grayscale
gray_image = cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)

# Calculate the average pixel intensity
average_intensity = np.mean(gray_image)
print('Average pixel intensity: {}'.format(average_intensity))

# Calculate the maximum pixel intensity
max_intensity = np.max(gray_image)
print('Maximum pixel intensity: {}'.format(max_intensity))

# Calculate the minimum pixel intensity
min_intensity = np.min(gray_image)
print('Minimum pixel intensity: {}'.format(min_intensity))


# In[12]:


# Convert the image from BGR to RGB format
blue = cv2.cvtColor(blue, cv2.COLOR_BGR2RGB)
# Extract the color channels
red_values = blue[:, :, 0].ravel()
green_values = blue[:, :, 1].ravel()
blue_values = blue[:, :, 2].ravel()

# Create histogram data
red_hist, _ = np.histogram(red_values, bins=256, range=(0, 256))
green_hist, _ = np.histogram(green_values, bins=256, range=(0, 256))
blue_hist, _ = np.histogram(blue_values, bins=256, range=(0, 256))

# Generate bin centers
bin_centers = np.arange(256)

# Plot the color distribution as lines
plt.plot(bin_centers, red_hist, color='red', label='Red')
plt.plot(bin_centers, green_hist, color='green', label='Green')
plt.plot(bin_centers, blue_hist, color='blue', label='Blue')

# Set plot title and labels
plt.title('Color Distribution of Blue Image')
plt.xlabel('Color Value')
plt.ylabel('Pixel Count')

# Add legend
plt.legend()

# Display the plot
plt.show()


# In[13]:


# Convert the image from BGR to RGB format
red = cv2.cvtColor(red, cv2.COLOR_BGR2RGB)

# Extract the color channels
red_values = red[:, :, 0].ravel()
green_values = red[:, :, 1].ravel()
blue_values = red[:, :, 2].ravel()

# Create histogram data
red_hist, _ = np.histogram(red_values, bins=256, range=(0, 256))
green_hist, _ = np.histogram(green_values, bins=256, range=(0, 256))
blue_hist, _ = np.histogram(blue_values, bins=256, range=(0, 256))

# Generate bin centers
bin_centers = np.arange(256)

# Plot the color distribution as lines
plt.plot(bin_centers, red_hist, color='red', label='Red')
plt.plot(bin_centers, green_hist, color='green', label='Green')
plt.plot(bin_centers, blue_hist, color='blue', label='Blue')

# Set plot title and labels
plt.title('Color Distribution of Red Image')
plt.xlabel('Color Value')
plt.ylabel('Pixel Count')

# Add legend
plt.legend()

# Display the plot


# In[14]:


# Convert the image from BGR to RGB format
orange = cv2.cvtColor(orange, cv2.COLOR_BGR2RGB)

# Extract the color channels
red_values = orange[:, :, 0].ravel()
green_values = orange[:, :, 1].ravel()
blue_values = orange[:, :, 2].ravel()

# Create histogram data
red_hist, _ = np.histogram(red_values, bins=256, range=(0, 256))
green_hist, _ = np.histogram(green_values, bins=256, range=(0, 256))
blue_hist, _ = np.histogram(blue_values, bins=256, range=(0, 256))

# Generate bin centers
bin_centers = np.arange(256)

# Plot the color distribution as lines
plt.plot(bin_centers, red_hist, color='red', label='Red')
plt.plot(bin_centers, green_hist, color='green', label='Green')
plt.plot(bin_centers, blue_hist, color='blue', label='Blue')

# Set plot title and labels
plt.title('Color Distribution of Orange Image')
plt.xlabel('Color Value')
plt.ylabel('Pixel Count')

# Add legend
plt.legend()

# Display the plot


# In[15]:


# Convert the image from BGR to RGB format
turquoise = cv2.cvtColor(turquoise, cv2.COLOR_BGR2RGB)

# Extract the color channels
red_values = turquoise[:, :, 0].ravel()
green_values = turquoise[:, :, 1].ravel()
blue_values = turquoise[:, :, 2].ravel()

# Create histogram data
red_hist, _ = np.histogram(red_values, bins=256, range=(0, 256))
green_hist, _ = np.histogram(green_values, bins=256, range=(0, 256))
blue_hist, _ = np.histogram(blue_values, bins=256, range=(0, 256))

# Generate bin centers
bin_centers = np.arange(256)

# Plot the color distribution as lines
plt.plot(bin_centers, red_hist, color='red', label='Red')
plt.plot(bin_centers, green_hist, color='green', label='Green')
plt.plot(bin_centers, blue_hist, color='blue', label='Blue')

# Set plot title and labels
plt.title('Color Distribution of Turquoise Image')
plt.xlabel('Color Value')
plt.ylabel('Pixel Count')

# Add legend
plt.legend()

# Display the plot


# In[16]:


# Convert the image from BGR to RGB format
sage = cv2.cvtColor(sage, cv2.COLOR_BGR2RGB)

# Extract the color channels
red_values = sage[:, :, 0].ravel()
green_values = sage[:, :, 1].ravel()
blue_values = sage[:, :, 2].ravel()

# Create histogram data
red_hist, _ = np.histogram(red_values, bins=256, range=(0, 256))
green_hist, _ = np.histogram(green_values, bins=256, range=(0, 256))
blue_hist, _ = np.histogram(blue_values, bins=256, range=(0, 256))

# Generate bin centers
bin_centers = np.arange(256)

# Plot the color distribution as lines
plt.plot(bin_centers, red_hist, color='red', label='Red')
plt.plot(bin_centers, green_hist, color='green', label='Green')
plt.plot(bin_centers, blue_hist, color='blue', label='Blue')

# Set plot title and labels
plt.title('Color Distribution of Sage Image')
plt.xlabel('Color Value')
plt.ylabel('Pixel Count')

# Add legend
plt.legend()


# In[17]:


import matplotlib.pyplot as plt

# Create the subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Plot 
# Convert the image from BGR to RGB format
blue = cv2.cvtColor(blue, cv2.COLOR_BGR2RGB)
# Extract the color channels
red_values = blue[:, :, 0].ravel()
green_values = blue[:, :, 1].ravel()
blue_values = blue[:, :, 2].ravel()

# Create histogram data
red_hist, _ = np.histogram(red_values, bins=256, range=(0, 256))
green_hist, _ = np.histogram(green_values, bins=256, range=(0, 256))
blue_hist, _ = np.histogram(blue_values, bins=256, range=(0, 256))

# Generate bin centers
bin_centers = np.arange(256)

# Plot the color distribution as lines
axes[0].plot(bin_centers, red_hist, color='red', label='Red')
axes[0].plot(bin_centers, green_hist, color='green', label='Green')
axes[0].plot(bin_centers, blue_hist, color='blue', label='Blue')

# Set plot title and labels
axes[0].set_title('Color Distribution of Blue Image')
axes[0].set_xlabel('Color Value')
axes[0].set_ylabel('Pixel Count')
axes[0].legend()


# Convert the image from BGR to RGB format
red = cv2.cvtColor(red, cv2.COLOR_BGR2RGB)

# Extract the color channels
red_values = red[:, :, 0].ravel()
green_values = red[:, :, 1].ravel()
blue_values = red[:, :, 2].ravel()

# Create histogram data
red_hist, _ = np.histogram(red_values, bins=256, range=(0, 256))
green_hist, _ = np.histogram(green_values, bins=256, range=(0, 256))
blue_hist, _ = np.histogram(blue_values, bins=256, range=(0, 256))

# Generate bin centers
bin_centers = np.arange(256)

# Plot the color distribution as lines
axes[1].plot(bin_centers, red_hist, color='red', label='Red')
axes[1].plot(bin_centers, green_hist, color='green', label='Green')
axes[1].plot(bin_centers, blue_hist, color='blue', label='Blue')

# Set plot title and labels
axes[1].set_title('Color Distribution of Red Image')
axes[1].set_xlabel('Color Value')
axes[1].set_ylabel('Pixel Count')
axes[1].legend()

# Plot 3
# Convert the image from BGR to RGB format
orange = cv2.cvtColor(orange, cv2.COLOR_BGR2RGB)

# Extract the color channels
red_values = orange[:, :, 0].ravel()
green_values = orange[:, :, 1].ravel()
blue_values = orange[:, :, 2].ravel()

# Create histogram data
red_hist, _ = np.histogram(red_values, bins=256, range=(0, 256))
green_hist, _ = np.histogram(green_values, bins=256, range=(0, 256))
blue_hist, _ = np.histogram(blue_values, bins=256, range=(0, 256))

# Generate bin centers
bin_centers = np.arange(256)

# Plot the color distribution as lines
axes[2].plot(bin_centers, red_hist, color='red', label='Red')
axes[2].plot(bin_centers, green_hist, color='green', label='Green')
axes[2].plot(bin_centers, blue_hist, color='blue', label='Blue')

# Set plot title and labels
axes[2].set_title('Color Distribution of Orange Image')
axes[2].set_xlabel('Color Value')
axes[2].set_ylabel('Pixel Count')
plt.legend()


# In[49]:


# Create the subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# Display the plot

# Plot 4
# Convert the image from BGR to RGB format
turquoise = cv2.cvtColor(turquoise, cv2.COLOR_BGR2RGB)

# Extract the color channels
red_values = turquoise[:, :, 0].ravel()
green_values = turquoise[:, :, 1].ravel()
blue_values = turquoise[:, :, 2].ravel()

# Create histogram data
red_hist, _ = np.histogram(red_values, bins=256, range=(0, 256))
green_hist, _ = np.histogram(green_values, bins=256, range=(0, 256))
blue_hist, _ = np.histogram(blue_values, bins=256, range=(0, 256))

# Generate bin centers
bin_centers = np.arange(256)

# Plot the color distribution as lines
axes[0].plot(bin_centers, red_hist, color='red', label='Red')
axes[0].plot(bin_centers, green_hist, color='green', label='Green')
axes[0].plot(bin_centers, blue_hist, color='blue', label='Blue')
plt.legend()
# Set plot title and labels
axes[0].set_title('Color Distribution of Turquoise Image')
axes[0].set_xlabel('Color Value')
axes[0].set_ylabel('Pixel Count')
axes[0].legend()


# Plot 5
# Convert the image from BGR to RGB format
sage = cv2.cvtColor(sage, cv2.COLOR_BGR2RGB)

# Extract the color channels
red_values = sage[:, :, 0].ravel()
green_values = sage[:, :, 1].ravel()
blue_values = sage[:, :, 2].ravel()

# Create histogram data
red_hist, _ = np.histogram(red_values, bins=256, range=(0, 256))
green_hist, _ = np.histogram(green_values, bins=256, range=(0, 256))
blue_hist, _ = np.histogram(blue_values, bins=256, range=(0, 256))

# Generate bin centers
bin_centers = np.arange(256)

# Plot the color distribution as lines
axes[1].plot(bin_centers, red_hist, color='red', label='Red')
axes[1].plot(bin_centers, green_hist, color='green', label='Green')
axes[1].plot(bin_centers, blue_hist, color='blue', label='Blue')

# Set plot title and labels
axes[1].set_title('Color Distribution of Sage Image')
axes[1].set_xlabel('Color Value')
axes[1].set_ylabel('Pixel Count')

# Add legend
plt.legend()



# Adjust the spacing between subplots
plt.tight_layout()

# Display the plots
plt.show()


# In[ ]:


# Divide the images by RGB channels and show the separated layers
for i, image in enumerate(images):
    # Split the image into color channels
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    # Display the separated layers
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(red_channel, cmap='Reds')
    axes[0].set_title('Red Channel')
    axes[0].axis('off')
    axes[1].imshow(green_channel, cmap='Greens')
    axes[1].set_title('Green Channel')
    axes[1].axis('off')
    axes[2].imshow(blue_channel, cmap='Blues')
    axes[2].set_title('Blue Channel')
    axes[2].axis('off')
    fig.suptitle(f'Image {i+1}: {image_paths[i]}', fontsize=14)
    plt.show()

