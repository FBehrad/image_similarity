import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from PIL import Image, ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the image preprocessing transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the pre-trained ResNet-50 model
model = resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # Remove the last fully connected layer
model = model.to(device)
model.eval()


class ImageDataset(Dataset):
    def __init__(self, folder_path, transform):
        self.image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image


def create_and_save_embeddings(model, dataloader, output_folder):
    with torch.no_grad():
        for batch_idx, images in enumerate(tqdm(dataloader)):
            images = images.to(device)
            embeddings = model(images)
            embeddings = F.normalize(embeddings.squeeze(), p=2, dim=1)  # Normalize the embeddings

            # Save embeddings to disk
            for idx, embedding in enumerate(embeddings):
                image_name = os.path.basename(dataloader.dataset.image_paths[batch_idx * dataloader.batch_size + idx])
                embedding_file = os.path.join(output_folder, f"{image_name[:-4]}.pt")
                torch.save(embedding, embedding_file)

            # Remove embeddings from memory
            del embeddings
            torch.cuda.empty_cache()


if __name__ == '__main__':
    folder1_path = r"D:\Datasets\PARA\train_imgs"
    folder2_path = r"D:\Datasets\PARA\test_imgs"


    dataset1 = ImageDataset(folder1_path, transform=transform)
    dataloader1 = DataLoader(dataset1, batch_size=512, shuffle=False, num_workers=0)
    dataset2 = ImageDataset(folder2_path, transform=transform)
    dataloader2 = DataLoader(dataset2, batch_size=512, shuffle=False, num_workers=0)


    output_folder_training = r"D:\Datasets\PARA\similarity\embeddings\train"
    output_folder_test = r"D:\Datasets\PARA\similarity\embeddings\test"
    os.makedirs(output_folder_training, exist_ok=True)
    os.makedirs(output_folder_test, exist_ok=True)

    create_and_save_embeddings(model, dataloader1, output_folder_training)
    create_and_save_embeddings(model, dataloader2, output_folder_test)

    print("Embeddings have been saved to disk.")
