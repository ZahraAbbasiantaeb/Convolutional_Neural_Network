import numpy as np

def read_all_images(path_to_data):

    with open(path_to_data, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)

        images = np.reshape(everything, (-1, 3, 96, 96))

        return images


def read_labels(path_to_labels):
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


TRAIN_DATA_PATH = 'data/train_X.bin'

TRAIN_LABEL_PATH = 'data/train_y.bin'

TEST_DATA_PATH = 'data/test_X.bin'

TEST_LABEL_PATH = 'data/test_y.bin'

train_data = read_all_images(TRAIN_DATA_PATH)
eval_data = read_all_images(TEST_DATA_PATH)

train_labels = read_labels(TRAIN_LABEL_PATH)
eval_labels = read_labels(TEST_LABEL_PATH)

train_data = train_data/np.float32(255)
train_labels = train_labels.astype(np.int32)  # not required

eval_data = eval_data/np.float32(255)
eval_labels = eval_labels.astype(np.int32)  # not required

for i in range(0, len(train_labels)):
    if(train_labels[i] == 10):
        train_labels[i] = 0

for i in range(0, len(eval_labels)):
    if(eval_labels[i]==10):
        eval_labels[i] = 0

print(set(train_labels))
