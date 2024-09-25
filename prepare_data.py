import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def normalize_position(position, mean_position, std_position):
    return (position - mean_position) / std_position

def normalize_rotation(quaternion):
    return quaternion / np.linalg.norm(quaternion)

# Load the JSON file
with open('new_animation_data.json') as f:
    animation_data = json.load(f)

# Organize the data into a list of frames and bones
all_frames = []
bone_names = list(animation_data[0]['bones'].keys())

# First pass: calculate mean position and standard deviation
all_positions = []
for frame_data in animation_data:
    for bone_name in bone_names:
        all_positions.append(frame_data['bones'][bone_name]['location'])

all_positions = np.array(all_positions)
mean_position = np.mean(all_positions, axis=0)
std_position = np.std(all_positions, axis=0)

# Avoid division by zero
std_position[std_position == 0] = 1

# Second pass: normalize data
for frame_data in animation_data:
    frame_info = []
    for bone_name in bone_names:
        bone_data = frame_data['bones'][bone_name]
        position = np.array(bone_data['location'])
        rotation = np.array(bone_data['rotation'])

        # Normalize position
        normalized_position = normalize_position(position, mean_position, std_position)
        # Normalize rotation (quaternion)
        normalized_rotation = normalize_rotation(rotation)
        # Combine normalized position and rotation into a single vector
        bone_vector = np.concatenate([normalized_position, normalized_rotation])
        frame_info.extend(bone_vector)

    all_frames.append(frame_info)

# Convert to numpy array for easier manipulation
all_frames = np.array(all_frames)

# Save the normalized data
np.save('all_frames_normalized.npy', all_frames)
np.save('bone_names.npy', np.array(bone_names))

# Save normalization parameters for later use
np.save('normalization_params.npy', {'mean_position': mean_position, 'std_position': std_position})

print("Data preparation complete. Saved all_frames_normalized.npy, bone_names.npy, and normalization_params.npy")
