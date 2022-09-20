import pyransac
import torch
import torchvision.transforms

from Training.models import Model
from Utilities.Visualize import Visualisation as vs
from Utilities.util import Utility
from PIL import Image


if __name__ == "__main__":
    model = Model.Model(num_features=2048, block_channel=[256, 512, 1024, 2048], pretrained=None)
    model = torch.nn.DataParallel(model).cuda()

    state_dict = torch.load("PEC-Hypersim.tar")['state_dict']
    model.load_state_dict(state_dict)

    transform = torchvision.transforms.ToTensor()
    ransac_params = pyransac.RansacParams(samples=3, iterations=2, confidence=0.98, threshold=0.5)

    with torch.no_grad():
        model.eval()

        image = transform(Image.open("Images/example.png")).unsqueeze(0)

        output_plane_para, output_embedding = model(image)
        output_normal = output_plane_para / torch.norm(output_plane_para, dim=1, keepdim=True)

        labels = Utility.embedding_segmentation(output_embedding)
        depth = torch.norm(output_plane_para, dim=1, keepdim=True)
        normals = Utility.aggregate_parameters_ransac(labels, output_normal, ransac_params)

        vs.image_show(image, "Original image")
        vs.image_show(labels, "Plane segmentation")
        vs.image_show(depth, "Depth estimation")
        vs.image_show(normals, "Normal estimation")

