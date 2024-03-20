import cv2
import torch
from torch.utils.data import DataLoader
import torchvision as tv
import os
import numpy as np
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
            for _, img in enumerate(image_aligned):
                aligned_faces.append(img[None])
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
    vid = cv2.VideoCapture(cam)
    placeholder = st.empty()  # Create a placeholder for the image

    while vid.grab():
        (
            _,
            img,
        ) = vid.retrieve()
        batch_boxes, aligned_images = mtcnn.detect_box(img)

        if aligned_images is not None:
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

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    img,
                    idx,
                    (x1 + 5, y1 + 10),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        placeholder.image(img, channels="RGB", use_column_width=True)


import streamlit as st
from PIL import Image


# Function to handle image upload and saving
def upload_image():
    uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "png", "webp"])
    name = st.text_input("Enter your name")

    if not os.path.exists(f"faces/{name}"):
        os.makedirs(f"faces/{name}")

    if uploaded_file is not None and name:
        image = Image.open(uploaded_file)
        image.save(os.path.join(f"faces/{name}", f"1.jpg"))
        st.success(f"Image saved as faces/{name}/1.jpg")
        return True
    return False


def save_image(img, name):
    if img is None or not name:
        return

    if not os.path.exists(f"faces/{name}"):
        os.makedirs(f"faces/{name}")

    image_path = os.path.join(f"faces/{name}", "1.jpg")
    cv2.imwrite(image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    st.success(f"Image saved as {image_path}")


def main():
    if not os.path.exists("faces"):
        os.makedirs("faces")

    st.title("FACE RECOGNITION")
    run_detection = st.button("Run Face Detection")

    if not run_detection:
        # Option to capture an image from the webcam or upload an image
        image_source = st.radio(
            "Select image source", ("Upload image", "Capture from webcam")
        )

        name = st.text_input("Enter your name", key="input_name")

        if image_source == "Upload image":
            uploaded_file = st.file_uploader(
                "Choose a photo", type=["jpg", "png", "webp"], key="image_uploader"
            )
            if uploaded_file is not None and name:
                img = np.array(Image.open(uploaded_file))
                st.image(img, channels="RGB", use_column_width=True)
                save_image(img, name)

        elif image_source == "Capture from webcam" and not run_detection:
            vid = cv2.VideoCapture(0)
            _, img_bgr = vid.read()
            vid.release()
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if name:
                st.image(img_rgb, channels="RGB", use_column_width=True)
                save_image(img_rgb, name)
    else:
        st.write("Images added, running face recognition!")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        mtcnn = get_mtcnn(device)
        resnet = get_resnet(device)

        embeddings, labels, idx_to_class = get_embeddings(
            faces_path="./faces",
            device=device,
            mtcnn=mtcnn,
            resnet=resnet,
            overwrite=True,
        )

        detect(
            device=device,
            mtcnn=mtcnn,
            resnet=resnet,
            embeddings=embeddings,
            labels=labels,
            idx_to_class=idx_to_class,
        )


if __name__ == "__main__":
    main()
