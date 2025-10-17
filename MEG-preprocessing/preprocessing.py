import os
import mne
import numpy as np
import pickle
import glob
import pandas as pd

def save_data(data, output_file):
    with open(output_file, 'wb') as file:
        pickle.dump(data, file, protocol=4)

fif_file = "/media/siat/disk1/BCI_data/THINGS-MEG/derivatives/preprocessed/preprocessed_P1-epo.fif"
output_dir = "/media/siat/disk1/BCI_data/THINGS-MEG/derivatives/preprocessed_divided"
def read_and_crop_epochs(fif_file):
    epochs = mne.read_epochs(fif_file, preload=True)
    cropped_epochs = epochs.crop(tmin=0, tmax=1.0)
    return cropped_epochs

epochs = read_and_crop_epochs(fif_file)    

sorted_indices = np.argsort(epochs.events[:, 2])
epochs = epochs[sorted_indices]

print('The epochs events is:', len(epochs.events))


csv_file_path = '/media/siat/disk1/BCI_data/THINGS/image-concept-index.csv'
image_concept_df = pd.read_csv(csv_file_path, header=None)
print('The image_concept_df is: ', image_concept_df)

# Accessing a column by its name
# Display the first few rows to understand its structure
print('The image_concept_df.shape[0] is: ', image_concept_df.shape[0])


def filter_valid_epochs(epochs, exclude_event_id=999999):
    return epochs[epochs.events[:, 2] != exclude_event_id]

valid_epochs = filter_valid_epochs(epochs)
print('The valid_epochs.info is: ', valid_epochs.info)
print('The valid_epochs.events.shape is:', valid_epochs.events.shape)


def identify_zs_event_ids(epochs, num_repetitions=12):
    event_ids = epochs.events[:, 2]
    unique_event_ids, counts = np.unique(event_ids, return_counts=True)
    zs_event_ids = unique_event_ids[counts == num_repetitions]
    return zs_event_ids
zs_event_ids = identify_zs_event_ids(valid_epochs)


# Separate and process datasets
training_epochs = valid_epochs[~np.isin(valid_epochs.events[:, 2], zs_event_ids)]
# Verify the number of events in the training set
print("Number of events in the training set:", len(training_epochs.events))


# Extract event IDs from the filtered training epochs
training_event_ids = np.unique(training_epochs.events[:, 2])

# Check for any overlap between zero-shot and training event IDs
overlap_ids = np.intersect1d(zs_event_ids, training_event_ids)

# Print the overlap, if any
print("Overlapping Event IDs:", overlap_ids)

zs_test_epochs = valid_epochs[np.isin(valid_epochs.events[:, 2], zs_event_ids)]
print('The zs_test_epochs.events is: ', zs_test_epochs.events)
print('The length of zs_test_epochs.events is: ', len(zs_test_epochs.events))
# zs_test_epochs.events

print('The length of training_epochs.events is: ', len(training_epochs.events))
print('The length of zs_test_epochs.events is:', len(zs_test_epochs.events))


training_event_ids = training_epochs.events[:, -1]
test_event_ids = zs_test_epochs.events[:, -1]

counts = {test_id: np.sum(training_event_ids == test_id) for test_id in test_event_ids}

# Assuming zs_event_ids is a numpy array or a list of event IDs
# Assuming image_concept_df is a pandas DataFrame with one column '1' representing image category indices

zs_event_to_category_map = {}

for i, event_id in enumerate(zs_event_ids):
    # Using the row index (i) to map to the image category index
    # Assuming the first event_id corresponds to the first row, second event_id to the second row, and so on
    image_category_index = image_concept_df.iloc[event_id-1, 0]  # Accessing the first (and only) column at row i
    zs_event_to_category_map[event_id] = image_category_index


# List to hold all the categories in the test set
test_set_categories = []

# Iterate over the event IDs in the test set
for event_id in zs_event_ids:
    if event_id in zs_event_to_category_map:
        # Get the category index from the mapping
        category_index = zs_event_to_category_map[event_id]
        test_set_categories.append(category_index)


from collections import Counter

# Count the occurrences of each category ID in the training set
category_counts = Counter(test_set_categories)


# Assuming zs_event_ids is a numpy array or a list of event IDs
# Assuming image_concept_df is a pandas DataFrame with one column '1' representing image category indices

event_to_category_map = {}

for i, event_id in enumerate(training_event_ids):
    # Using the row index (i) to map to the image category index
    # Assuming the first event_id corresponds to the first row, second event_id to the second row, and so on
    image_category_index = image_concept_df.iloc[event_id-1, 0]  # Accessing the first (and only) column at row i
    event_to_category_map[event_id] = image_category_index


# Assuming training_epochs is a variable that contains your training set epochs
# And it has an 'events' attribute similar to zs_test_epochs

# List to hold all the categories in the training set
train_set_categories = []

# Extract event IDs from the training set
training_event_ids = training_epochs.events[:, 2]

# Iterate over the event IDs in the training set
for event_id in training_event_ids:
    if event_id in event_to_category_map:
        # Get the category index from the mapping
        category_index = event_to_category_map[event_id]        
        train_set_categories.append(category_index)

from collections import Counter

# Count the occurrences of each category ID in the training set
category_counts = Counter(train_set_categories)



counts = {test_id: np.sum(train_set_categories == test_id) for test_id in test_set_categories}
# Calculate the total number of elements in 'counts'
total_elements = sum(counts.values())


# Assuming train_set_categories and test_set_categories are lists or numpy arrays

# Create a new list with elements from train_set_categories that are not in test_set_categories
train_set_categories_filtered = [item for item in train_set_categories if item not in test_set_categories]



# Create a mask for epochs to keep in the training set
keep_epochs_mask = [category not in test_set_categories for category in train_set_categories]
keep_epochs_mask
# Apply the mask to filter out epochs from training_epochs
training_epochs_filtered = training_epochs[keep_epochs_mask]

# Confirm the filtering
print("Original training set size:", len(training_epochs))
print("Filtered training set size:", len(training_epochs_filtered))

def reshape_meg_data(epochs, num_concepts, num_imgs, repetitions):
    data = epochs.get_data()
    reshaped_data = data.reshape((num_concepts, num_imgs, repetitions, data.shape[1], data.shape[2]))
    return reshaped_data


training_data = reshape_meg_data(training_epochs_filtered, num_concepts=1654, num_imgs=12, repetitions=1)
print('The training_data.shape is: ', training_data.shape)

zs_test_data = reshape_meg_data(zs_test_epochs, num_concepts=200, num_imgs=1, repetitions=12)
print('The zs_test_data.shape is: ', zs_test_data.shape)


import numpy as np
import os

def process_and_save_meg_data(fif_file, output_dir):
    epochs = read_and_crop_epochs(fif_file)
    
    sorted_indices = np.argsort(epochs.events[:, 2])
    epochs = epochs[sorted_indices]

    valid_epochs = filter_valid_epochs(epochs)
    zs_event_ids = identify_zs_event_ids(valid_epochs)

    training_epochs = valid_epochs[~np.isin(valid_epochs.events[:, 2], zs_event_ids)]
    zs_test_epochs = valid_epochs[np.isin(valid_epochs.events[:, 2], zs_event_ids)]

    keep_epochs_mask = [category not in test_set_categories for category in train_set_categories]
    training_epochs_filtered = training_epochs[keep_epochs_mask]

    training_data = reshape_meg_data(training_epochs_filtered, num_concepts=1654, num_imgs=12, repetitions=1)
    zs_test_data = reshape_meg_data(zs_test_epochs, num_concepts=200, num_imgs=1, repetitions=12)

    # Save data
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    save_data({'meg_data': training_data, 'ch_names': training_epochs_filtered.ch_names, 'times': training_epochs_filtered.times},
              os.path.join(output_dir, 'preprocessed_meg_training.pkl'))
    save_data({'meg_data': zs_test_data, 'ch_names': zs_test_epochs.ch_names, 'times': zs_test_epochs.times},
              os.path.join(output_dir, 'preprocessed_meg_zs_test.pkl'))

# fif_file = "/media/siat/disk1/BCI_data/THINGS-MEG/derivatives/preprocessed/preprocessed_P1-epo.fif"
# output_dir = "/media/siat/disk1/BCI_data/THINGS-MEG/derivatives/preprocessed_npy"
# process_and_save_meg_data(fif_file, output_dir)


def process_directory(input_dir, output_dir):
    fif_files = glob.glob(os.path.join(input_dir, '**/*epo.fif'), recursive=True)
    for fif_file in fif_files:
        filename = os.path.basename(fif_file)
        subject_num = filename.split('_')[1].split('-')[0]
        subject_dir_name = f"sub-{int(subject_num[1:]):02d}"
        subject_output_dir = os.path.join(output_dir, subject_dir_name)
        process_and_save_meg_data(fif_file, subject_output_dir)

in_dir = "/media/siat/disk1/BCI_data/THINGS-MEG/derivatives/preprocessed/"
output_dir = "/media/siat/disk1/BCI_data/THINGS-MEG/derivatives/preprocessed_divided"
process_directory(in_dir, output_dir)
