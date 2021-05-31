import numpy as np

kfolds = 5
images_left_out = 500
image_len = 5000
fold_interval = 1000

train_triplets = np.loadtxt("data/train_triplets.txt")

# target_fold_size = np.floor(train_triplets.shape[0] / kfolds)

"""images = train_triplets.flatten()
images, counts = np.unique(images, return_counts=True)

print(f"min count: {np.min(counts)}, max count: {np.max(counts)}")

current_fold_size = 0
current_fold_mask = np.ones(len(train_triplets), dtype=bool)

for i in range(len(counts)):
    count = counts[i]
    current_fold_size += count

    if current_fold_size > target_fold_size:
        break

    for triplet in train_triplets:
        pass"""

masks = np.ones((0, train_triplets.shape[0]))

for k in range(kfolds):
    mask = np.ones(train_triplets.shape[0])
    mask = mask.astype(bool)

    print(f"creating maks for image_left_out range: [{k * fold_interval}, {k * fold_interval + images_left_out}]")

    for i in range(train_triplets.shape[0]):
        if (k * fold_interval < train_triplets[i, 0]) and (train_triplets[i, 0] < k * fold_interval + images_left_out):
            mask[i] = False
        elif (k * fold_interval < train_triplets[i, 1]) and (train_triplets[i, 1] < k * fold_interval + images_left_out):
            mask[i] = False
        elif (k * fold_interval < train_triplets[i, 2]) and (train_triplets[i, 2] < k * fold_interval + images_left_out):
            mask[i] = False

    train_set = train_triplets[mask]
    validation_set = train_triplets[~mask]

    print(f"[fold {k}]: train_set shape: {train_set.shape}")
    print(f"[fold {k}]: validation_set shape: {validation_set.shape}")
    masks = np.vstack((masks, mask))

np.savetxt("data/fold_masks.txt", masks, fmt="%i")









