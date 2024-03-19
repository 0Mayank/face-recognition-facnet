import cv2
import torch
from torch.utils.data import DataLoader
import torchvision as tv
import os
import torch.nn.functional as F
from models import get_resnet, get_mtcnn


def encode_saved_images(
    path,
    device,
    embeddings_file,
    labels_file,
    idx_to_class_file,
    save_path,
    mtcnn,
    resnet,
    transform=None,
    workers=4,
):
    transform = tv.transforms.Compose([]) if transform is None else transform
    workers = 0 if os.name == "nt" else workers

    dataset = tv.datasets.ImageFolder(path, transform=transform)
    idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=lambda x: x[0], num_workers=workers)

    aligned_faces = []
    labels = []

    for image, label in loader:
        image_aligned = mtcnn(image)

        if image_aligned is not None:
            aligned_faces.append(image_aligned)
            labels.append(label)

    aligned_faces = torch.cat(aligned_faces, dim=0).to(device)
    embeddings = resnet(aligned_faces).detach().cpu()

    torch.save(embeddings, f"{save_path}/{embeddings_file}")
    torch.save(labels, f"{save_path}/{labels_file}")
    torch.save(idx_to_class, f"{save_path}/{idx_to_class_file}")

    return embeddings, labels, idx_to_class


def get_embeddings(
    faces_path,
    device,
    mtcnn,
    resnet,
    save_path="./data",
    embeddings_file="embeddings.pt",
    labels_file="labels.pt",
    transform=None,
    overwrite=False,
    idx_to_class_file="idx_to_class.dict",
):
    if (
        not overwrite
        and os.path.exists(save_path + "/" + embeddings_file)
        and os.path.exists(save_path + "/" + labels_file)
        and os.path.exists(save_path + "/" + idx_to_class_file)
    ):
        embeddings = torch.load(save_path + "/" + embeddings_file)
        labels = torch.load(save_path + "/" + labels_file)
        idx_to_class = torch.load(save_path + "/" + idx_to_class_file)

        return embeddings, labels, idx_to_class

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    return encode_saved_images(
        faces_path,
        device,
        embeddings_file,
        labels_file,
        idx_to_class_file,
        save_path,
        mtcnn,
        resnet,
        transform,
    )


def classify_image(image, device, resnet, embeddings, labels, threshold):
    embedding = resnet(image.to(device)).detach().cpu().reshape(1, -1)

    similarity = F.cosine_similarity(x1=embeddings, x2=embedding).reshape(-1)
    max_index = similarity.argmax().item()

    return labels[max_index] if similarity[max_index] > threshold else None


def detect(
    device, mtcnn, resnet, embeddings, labels, idx_to_class, cam=0, threshold=0.6
):
    import time

    vid = cv2.VideoCapture(cam)

    while vid.grab():
        (
            _,
            img,
        ) = vid.retrieve()
        batch_boxes, aligned_images = mtcnn.detect_box(img)

        msg = ""

        if aligned_images is not None:
            msg += "Detected: "
            for box, aligned in zip(batch_boxes, aligned_images):
                aligned = torch.Tensor(aligned.unsqueeze(0))
                x1, y1, x2, y2 = [int(x) for x in box]

                idx = classify_image(
                    image=aligned,
                    device=device,
                    resnet=resnet,
                    embeddings=embeddings,
                    labels=labels,
                    threshold=threshold,
                )
                idx = idx_to_class[idx] if idx is not None else "Unknown"

                msg += f"{idx} at [({x1}, {y1}), ({x2}, {y2})"
        else:
            msg += "No faces detected"

        print(msg)
        time.sleep(0.5)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mtcnn = get_mtcnn(device)
    resnet = get_resnet(device)

    embeddings, labels, idx_to_class = get_embeddings(
        faces_path="./faces", device=device, mtcnn=mtcnn, resnet=resnet
    )

    detect(
        device=device,
        mtcnn=mtcnn,
        resnet=resnet,
        embeddings=embeddings,
        labels=labels,
        idx_to_class=idx_to_class,
    )
