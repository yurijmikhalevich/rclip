import os

# Create output directory if it doesn't exist
import cv2
import numpy as np
import random

# Create output directory if it doesn't exist
output_dir = './images nested directories/generated_images'
os.makedirs(output_dir, exist_ok=True)

# Number of images to generate
num_images = 10000
image_size = (512, 512)  # Image size

# List of random shapes and objects to add to the image
shapes = ['circle', 'rectangle', 'line']
words = ['Tree', 'Car', 'House', 'Sun', 'Sky']

def generate_image(index: int):
    # Create a blank white image
    img = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255

    # Draw random shapes
    for _ in range(random.randint(1, 5)):  # Random number of shapes
        shape = random.choice(shapes)
        
        if shape == 'circle':
            center = (random.randint(50, image_size[0]-50), random.randint(50, image_size[1]-50))
            radius = random.randint(20, 80)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            thickness = random.randint(1, 5)
            cv2.circle(img, center, radius, color, thickness)
        
        elif shape == 'rectangle':
            pt1 = (random.randint(0, image_size[0]//2), random.randint(0, image_size[1]//2))
            pt2 = (random.randint(image_size[0]//2, image_size[0]), random.randint(image_size[1]//2, image_size[1]))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            thickness = random.randint(1, 5)
            cv2.rectangle(img, pt1, pt2, color, thickness)

        elif shape == 'line':
            pt1 = (random.randint(0, image_size[0]), random.randint(0, image_size[1]))
            pt2 = (random.randint(0, image_size[0]), random.randint(0, image_size[1]))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            thickness = random.randint(1, 5)
            cv2.line(img, pt1, pt2, color, thickness)

    # Add random text (object names)
    for _ in range(random.randint(1, 3)):  # Random number of text objects
        text = random.choice(words)
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (random.randint(50, image_size[0]-150), random.randint(50, image_size[1]-50))
        font_scale = random.uniform(0.5, 1.5)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        thickness = random.randint(1, 3)
        cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

    # Save the image
    cv2.imwrite(os.path.join(output_dir, f'image_{index}.png'), img)

# Generate the images
for i in range(num_images):
    generate_image(i)

print(f'{num_images} meaningful images generated successfully in {output_dir} directory.')

