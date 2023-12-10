import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import json
from PIL import ImageDraw


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, 'images', opt.phase)
        self.dir_bbox = os.path.join(opt.dataroot, 'bbox', opt.phase)

        #self.AB_paths, self.bbox_paths = sorted(make_dataset(self.dir_AB, self.dir_bbox))
        self.AB_paths, self.bbox_paths = make_dataset(self.dir_AB, self.dir_bbox)
        self.AB_paths = sorted(self.AB_paths)
        self.bbox_paths = sorted(self.bbox_paths)

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def draw_bbox(image, bbox, color='red', width=3):
        """Draw a bounding box on the image."""
        draw = ImageDraw.Draw(image)
        y, x, w, h = bbox
        draw.rectangle(((x, y), (x + w, y + h)), outline=color, width=width)
        return image   

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        bbox_path = self.bbox_paths[index]

        w_total = self.opt.loadSize * 2
        w = int(w_total / 2)
        h = self.opt.loadSize
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        # Attempt to load the bbox JSON file
        try:
           with open(bbox_path, 'r') as file:
              bbox = json.load(file)
        except Exception as e:
           raise RuntimeError(f"Error loading JSON file at {bbox_path}: {e}")

        # Confirm that the bbox data is a dictionary
        if not isinstance(bbox, dict):
           raise TypeError(f"Bounding box data is not a dictionary at index {index}: {bbox}")

    # Extract the bounding box values
        try:
           bbox_y = int(bbox.get('y', 0))
           bbox_x = int(bbox.get('x', 0))
           bbox_w = int(bbox.get('w', 0))
           bbox_h = int(bbox.get('h', 0))
        except ValueError as e:
           raise ValueError(f"Error processing bbox values at index {index}: {e}")

        """print(f"haha\nDataset bbox (index {index}): [{bbox_y}, {bbox_x}, {bbox_w}, {bbox_h}]")  # Debugging print"""

        bbox_x = max(int((bbox_x / self.opt.fineSize) * self.opt.loadSize), 0)
        bbox_y = max(int((bbox_y / self.opt.fineSize) * self.opt.loadSize), 0)
        bbox_w = max(int((bbox_w / self.opt.fineSize) * self.opt.loadSize), 0)
        bbox_h = max(int((bbox_h / self.opt.fineSize) * self.opt.loadSize), 0)

        if bbox_y <= h_offset or bbox_x <= w_offset:
            AB = Image.open(AB_path).convert('RGB')
            AB = AB.resize((self.opt.fineSize * 2, self.opt.fineSize), Image.BICUBIC)
            AB = self.transform(AB)
            A = AB[:, :self.opt.fineSize, :self.opt.fineSize]
            B = AB[:, :self.opt.fineSize, self.opt.fineSize:2 * self.opt.fineSize]
        else:
            AB = Image.open(AB_path).convert('RGB')
            AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
            AB = self.transform(AB)
            A = AB[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
            B = AB[:, h_offset:h_offset + self.opt.fineSize, w + w_offset:w + w_offset + self.opt.fineSize]

        return {'A': A, 'B': B, 'bbox': [bbox_y, bbox_x, bbox_w, bbox_h], 'A_paths': AB_path, 'B_paths': AB_path}


        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            #print A.size(2)
            bbox = [bbox[0], A.size(2) - bbox[2], A.size(2) - bbox[1], bbox[3]]
            #print('hehe')
            #print(bbox)
            #print(A.size())
            #print(bbox)
        return {'A': A, 'B': B, 'bbox': bbox,
                'A_paths': AB_path, 'B_paths': AB_path}
            

     

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
