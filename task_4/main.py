# import numpy as np
import numpy as np
import torch
import torchvision.models
from torchvision import datasets, transforms
# import matplotlib.pyplot as plt


# config
n_images = 10000
img_size = (224, 224)
image_loader_batch_size = 4
triplet_loader_batch_size = 32
encoder_features = 1000  # dependant on output of classifier

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

# Model
classifier_model = torchvision.models.mobilenet_v3_large(pretrained=True, progress=True)
classifier_model.eval()
"""device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Device: {device}")"""

# Classification prediction
# predictions = np.zeros((n_images, encoder_features))
"""for image_batch in imageloader:

    iter = 0
    for image in image_batch:
        print(f"type of image: {type(image)}")
        print(f"iter: {iter}")
        print(f"type of image_batch = {type(image)}")
        prediction = classifier_model(image)
        print(f"prediction.shape = {prediction.shape}")
        iter = iter + 1"""

for image in images:
    print(image[0])
    prediction = classifier_model(image[0])

print("finished")


"""images, labels = next(iter(dataloader))
# plt.imshow(np.squeeze(images[0].permute(1,2,0)))
plt.imshow(images[0])
plt.show()"""





