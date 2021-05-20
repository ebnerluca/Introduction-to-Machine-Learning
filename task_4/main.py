# import numpy as np
import numpy as np
import torch
import torchvision.models
from torchvision import datasets, transforms
# import matplotlib.pyplot as plt


#### Config ####
n_images = 10000
img_size = (224, 224)
image_loader_batch_size = 32
triplet_loader_batch_size = 32
encoder_features = 1000  # dependant on output of classifier

compute_features = True
features_path = "data/features.txt"
################


# Read Data
transform = transforms.Compose([transforms.Resize(img_size),
                               transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                           std=[0.229, 0.224, 0.225])])
images = datasets.ImageFolder("data", transform=transform)

triplets = np.loadtxt("data/train_triplets.txt")
print(f"triplets shape: {triplets.shape}")

# Data Loader
imageloader = torch.utils.data.DataLoader(images, batch_size=image_loader_batch_size, shuffle=False)
tripletloader = torch.utils.data.DataLoader(triplets, batch_size=triplet_loader_batch_size, shuffle=False)

# Model for feature predictions
classifier_model = torchvision.models.mobilenet_v3_small(pretrained=True, progress=True)
classifier_model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classifier_model.to(device)
print(f"Device: {device}")

# Compute features by using pretrained model
if compute_features:
    features = np.zeros((0, encoder_features))
    iter = 0
    for image_batch in imageloader:
        print(f"Computing features... {iter} / {n_images}", end="\r")#, flush=True)
        image_batch = image_batch[0]  # get rid of target classes
        image_batch = image_batch.to(device)
        features_batch = classifier_model(image_batch)
        features = np.vstack((features, features_batch.cpu().detach().numpy()))
        iter = iter + image_loader_batch_size
    print(f"\nComputing features done. Saving features under {str(features_path)}")
    np.savetxt(features_path, features)
else:
    print("Loading features...")
    features = np.loadtxt(features_path)
    print("Loading features done.")

print("All finished")






