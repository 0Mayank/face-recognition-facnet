# Face Recognition with FaceNet

This repository contains code for face recognition using the FaceNet model. The `faces` folder should be populated with subfolders, each representing a class (person) and containing photos of that person. The `main.py` script utilizes the FaceNet model to perform face recognition.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
  - [Folder Structure](#folder-structure)
  - [Running the Code](#running-the-code)
- [Working Explanation](#working-explanation)

## Introduction

This project focuses on face recognition using the FaceNet model. The model is trained to map faces into a high-dimensional space and measure cosine similarity between them, making it robust for face recognition tasks.

## Setup

### Folder Structure

Ensure that you have the following folder structure in place:

```plaintext
- faces/
  - class1/
    - photo1.jpg
    - photo2.jpg
    ...
  - class2/
    - photo1.jpg
    - photo2.jpg
    ...
  ...
```

- `faces/`: Main folder containing subfolders for each class (person).
- `class1/`, `class2/`: Subfolders for individual classes, each containing photos of the respective person.

### Running the Code

1. Create the required folder structure with photos.
2. Run the `main.py` script.

```bash
python main.py
```

The script will use the FaceNet model to perform face recognition on the provided photos.

## Working Explanation

For a detailed explanation of the working of the FaceNet model and the code, refer to the `facenet.ipynb` Jupyter notebook.
