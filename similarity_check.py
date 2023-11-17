import os
import torch
from tqdm import tqdm
import shutil
import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description="Finding similar images among two data folders")
    parser.add_argument(
        '--first_folder', help="the address of the first folder")
    parser.add_argument(
        '--second_folder', help="the address of the second folder")

    parser.add_argument(
        '--device', help="device",
        default='cuda:0')
    parser.add_argument(
        '--threshold', help="the higher the stricter", default='0.89')
    parser.add_argument(
        '--batch_size', help="batch size", default='1024')
    parser.add_argument('--output_folder', help='the place where the similar images are saved', default=r'outputs')
    arguments = parser.parse_args()
    return arguments


def load_embedding(embedding_path):
    return torch.load(embedding_path)


def read_embeddings_from_folder(embedding_folder):
    embeddings = {}
    for filename in os.listdir(embedding_folder):
        if filename.endswith(".pt"):
            embedding_path = os.path.join(embedding_folder, filename)
            image_name = os.path.splitext(filename)[0]
            embedding = load_embedding(embedding_path)
            embeddings[image_name] = embedding
    return embeddings


def batch_cosine_similarity(embeddings1, embeddings2):
    # Normalize the embeddings along the last dimension
    normalized_embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=-1)
    normalized_embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=-1)

    # Compute the dot product between normalized embeddings
    dot_product = torch.matmul(normalized_embeddings1, normalized_embeddings2.transpose(-2, -1))

    return dot_product


def find_similar_embeddings(embeddings_folder1, embeddings_folder2, folder1, folder2, output, batch_size=64,
                            threshold=0.9):
    numbers = 0
    paths_1 = os.listdir(embeddings_folder1)
    paths_2 = os.listdir(embeddings_folder2)

    batch_list1 = [paths_1[i:i + batch_size] for i in range(0, len(paths_1), batch_size)]
    batch_list2 = [paths_2[i:i + batch_size] for i in range(0, len(paths_2), batch_size)]

    for batch1 in tqdm(batch_list1):
        embeddings1 = []
        image_names1 = []
        for path in batch1:
            embed1 = load_embedding(os.path.join(embeddings_folder1, path))
            embeddings1.append(embed1)
            image_names1.append(os.path.splitext(path)[0])
        embeddings1 = torch.stack(embeddings1)
        for batch2 in batch_list2:
            embeddings2 = []
            image_names2 = []
            for path in batch2:
                embed2 = load_embedding(os.path.join(embeddings_folder2, path))
                embeddings2.append(embed2)
                image_names2.append(os.path.splitext(path)[0])
            embeddings2 = torch.stack(embeddings2)

            similarity_matrix = batch_cosine_similarity(embeddings1, embeddings2)

            # Find indices of similar embeddings based on the threshold
            row_indices, col_indices = torch.nonzero(similarity_matrix > threshold, as_tuple=True)

            number = len(row_indices)
            numbers += number
            for i in range(number):
                id, id2 = row_indices[i].item(), col_indices[i].item()
                image_name = image_names1[id]
                image_name_2 = image_names2[id2]

                similarity = similarity_matrix[id, id2].item()

                print(f'Image 1 : {image_name}.jpg \t Image 2 : {image_name_2}.jpg \t Similarity: {similarity}')

                name1 = f'{id}_{id2}_train_{image_name}.jpg'
                name2 = f'{id}_{id2}_test_{image_name_2}.jpg'

                shutil.copy(os.path.join(folder1, f'{image_name}.jpg'), os.path.join(output, name1))
                shutil.copy(os.path.join(folder2, f'{image_name_2}.jpg'), os.path.join(output, name2))

    print(f'With threshold: {threshold}, {numbers} similar images have been found.')


folder1 = r"D:\Datasets\PARA\train_imgs"
folder2 = r"D:\Datasets\PARA\test_imgs"
embedding_folder1 = r"D:\Datasets\PARA\similarity\embeddings\train"
embedding_folder2 = r"D:\Datasets\PARA\similarity\embeddings\test"


threshold = 0.89
batch_size = 1024
output = r"D:\Datasets\PARA\similarity\similar_images\0.89"
os.makedirs(output, exist_ok=True)


