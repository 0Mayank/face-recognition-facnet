FROM python:3.12-slim-bullseye

# Setup virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install opencv-python-headless facenet-pytorch

# Copy the required files
COPY faces ./faces
COPY models.py ./models.py
COPY main.py ./main.py

# Setup the command, use unbuffered output (-u)
CMD ["python", "-u", "main.py"]
