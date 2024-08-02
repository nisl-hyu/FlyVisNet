# Generate moving patterns dataset

# Angel Canelo 2024.07.19

import numpy as np
import random
from scipy.io import savemat
import cv2

# Constants
dataset = 0     # 0 -> 244x324 1 -> 20x40
traintest = 0   # 0 -> Train 1 -> Test

if dataset==0:
    IMG_WIDTH = 324
    IMG_HEIGHT = 244
    BARWMIN = 5
    BARWMAX = 20
    SPOTMIN = 5
    SPOTMAX = 20
    if traintest == 0:
        NUM_SEQUENCES = 100  # Number of sequences per label
        OUTPUT_DIR = '../data/data_moving_pattern_244x324_train.mat'
    elif traintest == 1:
        NUM_SEQUENCES = 10  # Number of sequences per label
        OUTPUT_DIR = '../data/data_moving_pattern_244x324_test.mat'
elif dataset==1:
    IMG_WIDTH = 40
    IMG_HEIGHT = 20
    BARWMIN = 1
    BARWMAX = 4
    SPOTMIN = 1
    SPOTMAX = 4
    if traintest == 0:
        NUM_SEQUENCES = 100  # Number of sequences per label
        OUTPUT_DIR = '../data/data_moving_pattern_20x40_train.mat'
    elif traintest == 1:
        NUM_SEQUENCES = 10  # Number of sequences per label
        OUTPUT_DIR = '../data/data_moving_pattern_20x40_test.mat'

NUM_FRAMES = 25  # Number of frames per sequence

LABELS = ['loom', 'bar', 'spot']

# Ensure output directories exist
#os.makedirs(OUTPUT_DIR, exist_ok=True)


def random_brightness():
    """Generate a random brightness value between 0.5 and 1.5"""
    return random.uniform(0, 1)


def generate_bar_sequence(brightness, direction):
    """Generate a sequence of images with a moving bar"""
    sequence = []
    seq_label = []
    centroids = []
    bar_width = random.randint(BARWMIN, BARWMAX)
    bar_height = int(IMG_HEIGHT*0.8)
    bar_color = int(255 * brightness)
    back_color = random_brightness()
    while int(255 * back_color) == bar_color:
        back_color = random_brightness()

    if direction == 'left_to_right':
        for frame in range(NUM_FRAMES):
            image = np.full((IMG_HEIGHT, IMG_WIDTH, 1), int(255 * back_color), dtype=np.uint8)
            bar_x = int((IMG_WIDTH - bar_width) * frame / (NUM_FRAMES - 1))
            bar_y = IMG_HEIGHT // 2
            cv2.rectangle(image, (bar_x, bar_y-bar_height//2), (bar_x + bar_width, bar_y + bar_height//2), (bar_color, bar_color, bar_color), -1)
            n_perc = 0.05  # noise strength
            noise = n_perc * np.random.randint(0, 256, image.shape, dtype=np.uint8)
            image = np.clip(image.astype(np.float32) + noise.astype(np.float32), 0, 255).astype(np.uint8)
            sequence.append(image)
            seq_label.append([0, 1, 0])
            centroids.append(bar_x + bar_width / 2)
            # plt.imshow(image, cmap='gray', vmin=0, vmax=255)
            # plt.pause(0.1)
    else:
        for frame in range(NUM_FRAMES):
            image = np.full((IMG_HEIGHT, IMG_WIDTH, 1), int(255 * back_color), dtype=np.uint8)
            bar_x = int((IMG_WIDTH - bar_width) * (1 - frame / (NUM_FRAMES - 1)))
            bar_y = IMG_HEIGHT // 2
            cv2.rectangle(image, (bar_x, bar_y-bar_height//2), (bar_x + bar_width, bar_y + bar_height//2), (bar_color, bar_color, bar_color), -1)
            n_perc = 0.05  # noise strength
            noise = n_perc * np.random.randint(0, 256, image.shape, dtype=np.uint8)
            image = np.clip(image.astype(np.float32) + noise.astype(np.float32), 0, 255).astype(np.uint8)
            sequence.append(image)
            seq_label.append([0, 1, 0])
            centroids.append(bar_x + bar_width / 2)
            # plt.imshow(image, cmap='gray', vmin=0, vmax=255)
            # plt.pause(0.1)

    return sequence, seq_label, centroids


def generate_spot_sequence(brightness, direction):
    """Generate a sequence of images with a moving spot (square)"""
    sequence = []
    seq_label = []
    centroids = []
    spot_size = random.randint(SPOTMIN, SPOTMAX)
    spot_color = int(255 * brightness)
    back_color = random_brightness()
    while int(255 * back_color) == spot_color:
        back_color = random_brightness()

    if direction == 'left_to_right':
        for frame in range(NUM_FRAMES):
            image = np.full((IMG_HEIGHT, IMG_WIDTH, 1), int(255 * back_color), dtype=np.uint8)
            spot_x = int((IMG_WIDTH - spot_size) * frame / (NUM_FRAMES - 1))
            spot_y = IMG_HEIGHT // 2 #random.randint(0, IMG_HEIGHT - spot_size)
            cv2.rectangle(image, (spot_x, spot_y - spot_size//2), (spot_x + spot_size, spot_y + spot_size//2),
                          (spot_color, spot_color, spot_color), -1)
            n_perc = 0.05  # noise strength
            noise = n_perc * np.random.randint(0, 256, image.shape, dtype=np.uint8)
            image = np.clip(image.astype(np.float32) + noise.astype(np.float32), 0, 255).astype(np.uint8)
            sequence.append(image)
            seq_label.append([0, 0, 1])
            centroids.append(spot_x + spot_size / 2)
            # plt.imshow(image, cmap='gray', vmin=0, vmax=255)
            # plt.pause(0.1)
    else:
        for frame in range(NUM_FRAMES):
            image = np.full((IMG_HEIGHT, IMG_WIDTH, 1), int(255 * back_color), dtype=np.uint8)
            spot_x = int((IMG_WIDTH - spot_size) * (1 - frame / (NUM_FRAMES - 1)))
            spot_y = IMG_HEIGHT // 2 #random.randint(0, IMG_HEIGHT - spot_size)
            cv2.rectangle(image, (spot_x, spot_y - spot_size//2), (spot_x + spot_size, spot_y + spot_size//2),
                          (spot_color, spot_color, spot_color), -1)
            n_perc = 0.05  # noise strength
            noise = n_perc * np.random.randint(0, 256, image.shape, dtype=np.uint8)
            image = np.clip(image.astype(np.float32) + noise.astype(np.float32), 0, 255).astype(np.uint8)
            sequence.append(image)
            seq_label.append([0, 0, 1])
            centroids.append(spot_x + spot_size / 2)
            # plt.imshow(image, cmap='gray', vmin=0, vmax=255)
            # plt.pause(0.1)

    return sequence, seq_label, centroids


def generate_loom_sequence(brightness):
    """Generate a sequence of images with an expanding square (loom)"""
    sequence = []
    seq_label = []
    centroids = []
    max_size = min(IMG_WIDTH, IMG_HEIGHT) // 2
    loom_color = int(255 * brightness)
    back_color = random_brightness()
    while int(255 * back_color) == loom_color:
        back_color = random_brightness()

    # Calculate the three predetermined x positions
    positions = [
        IMG_WIDTH // 4,       # Left center quarter
        IMG_WIDTH // 2,       # Center of the frame
        IMG_WIDTH * 3 // 4    # Right center quarter
    ]

    # Randomly select one of the three positions
    loom_center_x = np.random.choice(positions)
    loom_center_y = IMG_HEIGHT // 2

    for frame in range(NUM_FRAMES):
        image = np.full((IMG_HEIGHT, IMG_WIDTH, 1), int(255 * back_color), dtype=np.uint8)
        loom_size = int(max_size * frame / (NUM_FRAMES - 1))
        top_left = (loom_center_x - loom_size, loom_center_y - loom_size)
        bottom_right = (loom_center_x + loom_size, loom_center_y + loom_size)
        cv2.rectangle(image, top_left, bottom_right, (loom_color, loom_color, loom_color), -1)
        n_perc = 0.05  # noise strength
        noise = n_perc * np.random.randint(0, 256, image.shape, dtype=np.uint8)
        image = np.clip(image.astype(np.float32) + noise.astype(np.float32), 0, 255).astype(np.uint8)
        sequence.append(image)
        seq_label.append([1, 0, 0])
        centroids.append(loom_center_x)
        # plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        # plt.pause(0.1)

    return sequence, seq_label, centroids


# Generate sequences and store them in lists
# data = {label: [] for label in LABELS}
data = {'Images': [], 'Image_label': [], 'X': []}

for label in LABELS:
    for ii in range(NUM_SEQUENCES):
        object_brightness = random_brightness()
        if label == 'bar':
            direction = random.choice(['left_to_right', 'right_to_left'])
            sequence, seq_label, centroids = generate_bar_sequence(object_brightness, direction)
            #imageio.mimsave(f"Bar_{ii}.gif", np.array(sequence), duration=0.1)
        elif label == 'spot':
            direction = random.choice(['left_to_right', 'right_to_left'])
            sequence, seq_label, centroids = generate_spot_sequence(object_brightness, direction)
            #imageio.mimsave(f"Spot_{ii}.gif", np.array(sequence), duration=0.1)
        elif label == 'loom':
            sequence, seq_label, centroids = generate_loom_sequence(object_brightness)
            #imageio.mimsave(f"Loom_{ii}.gif", np.array(sequence), duration=0.1)
        data['Images'].extend(sequence)
        data['Image_label'].extend(seq_label)
        data['X'].extend(centroids)

# Save to .mat file
print(len(data['Images']))
savemat(OUTPUT_DIR, data)