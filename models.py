from facenet_pytorch import MTCNN, InceptionResnetV1
from types import MethodType


def get_resnet(device):
    return InceptionResnetV1(pretrained='vggface2').eval().to(device)


def get_mtcnn(device):
    mtcnn = MTCNN(keep_all=True, device=device)

    def detect_box(self, img, save_path=None):
        # Detect faces
        batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
        # Select faces
        if not self.keep_all:
            batch_boxes, batch_probs, batch_points = self.select_boxes(
                batch_boxes, batch_probs, batch_points, img, method=self.selection_method
            )
        # Extract faces
        faces = self.extract(img, batch_boxes, save_path)
        return batch_boxes, faces

    mtcnn.detect_box = MethodType(detect_box, mtcnn)
    return mtcnn
