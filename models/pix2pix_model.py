import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from copy import deepcopy
from . import networks
from PIL import Image
import torchvision.transforms as transforms
from pg_modules.discriminator import ProjectedDiscriminator
import torch.nn.functional as F
import torchvision.utils as vutils



class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):

        BaseModel.initialize(self, opt)
        # self.opt = opt
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)
        self.netD_person = ProjectedDiscriminator()

        if torch.cuda.is_available():
         self.netD_person.cuda()


        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids)
        self.netG.cuda()
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_image = networks.define_image_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
            use_sigmoid = not opt.no_lsgan
            #self.netD_person = networks.define_person_D(opt.input_nc, opt.ndf, opt, use_sigmoid, self.gpu_ids)
            self.netD_image.cuda()

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD_image, 'D_image', opt.which_epoch)
                self.load_network(self.netD_person, 'D_person', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            gan_mode = 'vanilla' if opt.no_lsgan else 'lsgan'
            #print('haha'+ str(opt.no_lsgan))
            # self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            #self.criterionGAN_image = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            #self.criterionGAN_person = networks.GANLoss(use_lsgan=opt.no_lsgan, tensor=self.Tensor)
            self.criterionGAN_image = networks.GANLoss(gan_mode=gan_mode)
            self.criterionGAN_person = networks.GANLoss(gan_mode=gan_mode)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_image = torch.optim.Adam(self.netD_image.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_person = torch.optim.Adam(self.netD_person.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD_image)
            networks.print_network(self.netD_person)
        print('-----------------------------------------------')
        self.save_images_now = False  # Set the default value to False
        self.person_crop_real = None
        self.person_crop_fake = None

    def set_input(self, input):
        if torch.cuda.is_available():
            self.input_A = self.input_A.cuda()
            self.input_B = self.input_B.cuda()

        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        # Debugging print for received bbox data
        """print("Received bbox data:", input['bbox'])"""

        # Unpack the batched bbox data
        batched_bbox_data = input['bbox']
        self.bbox = []
        for i in range(len(batched_bbox_data[0])):
            bbox = [batched_bbox_data[j][i].item() for j in range(4)]
            self.bbox.append(bbox)

        # Debugging print for processed bbox structure
        """print("Debug - Processed bbox structure:", self.bbox)"""


    def denormalize(self, tensor):
      #"""Denormalizes a tensor from [-1,1] range to [0,1] range"""
        return (tensor + 1) / 2   

    def forward(self):
         self.real_A = Variable(self.input_A)
         self.fake_B = self.netG.forward(self.real_A)
         self.real_B = Variable(self.input_B)

         _, _, height, width = self.real_B.size()

         batch_size, channels, _, _ = self.real_B.size()
         # Initialize person_crop_real and person_crop_fake tensors with -1 (or any other value as needed)
         self.person_crop_real = torch.full_like(self.real_B, fill_value=-1)
         self.person_crop_fake = torch.full_like(self.fake_B, fill_value=-1)
         #self.person_crop_real = torch.zeros_like(self.real_B)
         #self.person_crop_fake = torch.zeros_like(self.fake_B)

         for i in range(batch_size):
             if len(self.bbox[i]) == 4:
                 y, x, w, h = self.bbox[i]

                 # Calculate actual width and height of the bbox
                 bbox_width = min(w - x, width - x)
                 bbox_height = min(h - y, height - y)

                 # Crop the image using the adjusted bbox coordinates
                 cropped_real = self.real_B[i, :, y:y + bbox_height, x:x + bbox_width]
                 cropped_fake = self.fake_B[i, :, y:y + bbox_height, x:x + bbox_width]

                 # Assign the cropped images to person_crop_real and person_crop_fake
                 self.person_crop_real[i, :, :bbox_height, :bbox_width] = cropped_real
                 self.person_crop_fake[i, :, :bbox_height, :bbox_width] = cropped_fake
             else:
                 print(f"Warning - Incorrect bbox format for image index {i}: {self.bbox[i]}")

         """os.makedirs('/content/cropped_images', exist_ok=True)  # Modify this path if not using Google Colab
         for i in range(min(3, self.person_crop_real.size(0))):
          # Denormalize and save images
            real_A_denorm = self.denormalize(self.real_A[i])
            real_B_denorm = self.denormalize(self.real_B[i])
            person_crop_real_denorm = self.denormalize(self.person_crop_real[i])
            person_crop_fake_denorm = self.denormalize(self.person_crop_fake[i])

            vutils.save_image(real_A_denorm, f'/content/cropped_images/real_A{i}.png')
            vutils.save_image(real_B_denorm, f'/content/cropped_images/real_B{i}.png')
            vutils.save_image(person_crop_real_denorm, f'/content/cropped_images/person_crop_real_{i}.png')
            vutils.save_image(person_crop_fake_denorm, f'/content/cropped_images/person_crop_fake_{i}.png')"""
            

    """def test(self):
        super(Pix2PixModel, self).eval()

        with torch.no_grad():
            self.real_A = Variable(self.input_A)
            self.fake_B = self.netG.forward(self.real_A)
            self.real_B = Variable(self.input_B)

            y, x, w, h = self.bbox
            self.person_crop_real = self.real_B[:,:,y[0]:h[0],x[0]:w[0]]
            self.person_crop_fake = self.fake_B[:,:,y[0]:h[0],x[0]:w[0]]
        #self.real_A = Variable(self.input_A, volatile=True)
        #self.fake_B = self.netG.forward(self.real_A)
        #self.real_B = Variable(self.input_B, volatile=True)

        #y,x,w,h = self.bbox
        #self.person_crop_real = self.real_B[:,:,y[0]:h[0],x[0]:w[0]]
        #self.person_crop_fake = self.fake_B[:,:,y[0]:h[0],x[0]:w[0]]

    # get image paths
    """
    def test(self):
        super(Pix2PixModel, self).eval()

        with torch.no_grad():
           self.real_A = Variable(self.input_A)
           self.fake_B = self.netG.forward(self.real_A)
           self.real_B = Variable(self.input_B)

           batch_size, _, height, width = self.real_B.size()
           self.person_crop_real = torch.zeros_like(self.real_B)
           self.person_crop_fake = torch.zeros_like(self.fake_B)

           for i in range(batch_size):
               if len(self.bbox[i]) == 4:
                   y, x, w, h = self.bbox[i]

                   # Ensure the bbox coordinates are within the image dimensions
                   y_end = min(y + h, height)
                   x_end = min(x + w, width)
                   # Crop the images within the bounding box dimensions
                   cropped_real = self.real_B[i, :, y:y_end, x:x_end]
                   cropped_fake = self.fake_B[i, :, y:y_end, x:x_end]

                  # Calculate the size of the cropped images
                   crop_height, crop_width = cropped_real.shape[1], cropped_real.shape[2]

                # Place the cropped images into the appropriately sized tensors
                   self.person_crop_real[i, :, :crop_height, :crop_width] = cropped_real
                   self.person_crop_fake[i, :, :crop_height, :crop_width] = cropped_fake
               else:
                   print(f"Warning: Incorrect bbox format for image index {i}: {self.bbox[i]}")

    def get_image_paths(self):
        return self.image_paths

    def backward_D_image(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        self.pred_fake = self.netD_image.forward(fake_AB.detach())
        # self.loss_D_image_fake = self.criterionGAN(self.pred_fake, False)
        self.loss_D_image_fake = self.criterionGAN_image(self.pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.pred_real = self.netD_image.forward(real_AB)
        # self.loss_D_image_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_image_real = self.criterionGAN_image(self.pred_real, True)

        # Combined loss
        self.loss_D_image = (self.loss_D_image_fake + self.loss_D_image_real) * 0.5

        self.loss_D_image.backward()

    def crop_consistency_loss(self, real, fake, bbox):
    #"""
    #Calculate the L1 loss between the cropped regions of the real and fake images.
     y, x, h, w = bbox
     real_crop = real[:, :, y:y+h, x:x+w]
     fake_crop = fake[:, :, y:y+h, x:x+w]
     return torch.nn.functional.l1_loss(real_crop, fake_crop)

    def resize_and_pad(self, image, target_size):
        _, _, h, w = image.size()
        # Determine the scaling factor and resize the image
        scale_factor = target_size / max(h, w)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # Calculate padding to make the image square
        padding_top = (target_size - new_h) // 2
        padding_bottom = target_size - new_h - padding_top
        padding_left = (target_size - new_w) // 2
        padding_right = target_size - new_w - padding_left

        # Apply padding
        #print("After resizing: ", image.size())
        image_padded = F.pad(image, (padding_left, padding_right, padding_top, padding_bottom), mode='constant', value=0)
        #print("After padding: ", image_padded.size())
        return image_padded 

    def backward_D_person(self):
        batch_size = 1
        c_dim = 1000
        dummy_c = torch.zeros(batch_size, c_dim, device=self.person_crop_real.device)
        #print("Size of person_crop_real:", self.person_crop_real.size())
        #print("Size of person_crop_fake:", self.person_crop_fake.size())
        # Resize and pad crops to 224x224 (or your target size)
        target_size = 224
        self.person_crop_real = self.resize_and_pad(self.person_crop_real, target_size)
        self.person_crop_fake = self.resize_and_pad(self.person_crop_fake, target_size)

        # Now print the sizes after resizing and padding
        #print("Size of person_crop_real after resize and pad:", self.person_crop_real.size())
        #print("Size of person_crop_fake after resize and pad:", self.person_crop_fake.size())
        # Save the image pairs as they are presented to the person discriminator
        """os.makedirs('croppedimagesD', exist_ok=True)
        for i in range(min(5, self.person_crop_real.size(0))):
            real_image = self.denormalize(self.person_crop_real[i])
            fake_image = self.denormalize(self.person_crop_fake[i])
            vutils.save_image(real_image, f'croppedimagesD/person_crop_real_{i}.png')
            vutils.save_image(fake_image, f'croppedimagesD/person_crop_fake_{i}.png')"""

        # Compute loss using Projected GAN's netD_person
        pred_person_real = self.netD_person(self.person_crop_real, dummy_c)
        pred_person_fake = self.netD_person(self.person_crop_fake, dummy_c)

        # Calculate real and fake losses
        #self.loss_D_person_real = self.criterionGAN_person(pred_person_real, True)
        #self.loss_D_person_fake = self.criterionGAN_person(pred_person_fake, False)
        self.loss_D_person_real = F.relu(0.35 - pred_person_real).mean()
        self.loss_D_person_fake = F.relu(0.35 + pred_person_fake).mean()

        # Combine loss
        self.loss_D_person = (self.loss_D_person_fake + self.loss_D_person_real) * 0.5
        self.loss_D_person.backward()



    def backward_G(self):
        batch_size = 1  # Assuming batch size is 1; adjust if necessary

        # Initialize the losses
        self.loss_G_Crop_Consistency = 0
        self.loss_G_GAN_image = 0
        self.loss_G_L1 = 0
        self.loss_G_GAN_person = 0

        c_dim = 1000  # Assuming c_dim is 1000 or as per your discriminator's requirement
        dummy_c = torch.zeros(batch_size, c_dim, device=self.person_crop_real.device)

        for i in range(self.real_B.size(0)):  # Loop through each image in the batch
            if len(self.bbox[i]) == 4:
                y, x, w, h = self.bbox[i]

                # Calculate Crop Consistency Loss for each image
                self.loss_G_Crop_Consistency += self.crop_consistency_loss(self.real_B[i:i+1], self.fake_B[i:i+1], (y, x, h, w))

                # Process for GAN and L1 losses
                fake_AB = torch.cat((self.real_A[i:i+1], self.fake_B[i:i+1]), 1)
                pred_fake_image = self.netD_image.forward(fake_AB)
                self.loss_G_GAN_image = self.criterionGAN_image(pred_fake_image, True)

                self.loss_G_L1 = self.criterionL1(self.fake_B[i:i+1], self.real_B[i:i+1]) * self.opt.lambda_A

                pred_fake_person = self.netD_person.forward(self.person_crop_fake[i:i+1], dummy_c)
                self.loss_G_GAN_person = self.criterionGAN_person(pred_fake_person, True) * self.opt.lambda_C
            else:
                print(f"Warning - Incorrect bbox format for image index {i}: {self.bbox[i]}")

    # Normalize the losses by the number of images in the batch
        self.loss_G_Crop_Consistency /= self.real_B.size(0)
        self.loss_G_GAN_image /= self.real_B.size(0)
        self.loss_G_L1 /= self.real_B.size(0)
        self.loss_G_GAN_person /= self.real_B.size(0)

        # Combine with existing losses
        lambda_crop = 100  # Weight for crop consistency loss, can be tuned
        self.loss_G = self.loss_G_GAN_image + self.loss_G_L1 + self.loss_G_GAN_person + lambda_crop * self.loss_G_Crop_Consistency

        self.loss_G.backward()


    def get_current_errors(self):
        return OrderedDict([
        # ... existing errors ...
        ('G_Crop_Consistency', self.loss_G_Crop_Consistency.cpu().data)
        ])


    def optimize_parameters(self, only_d):

        self.forward()
        self.optimizer_D_image.zero_grad()
        self.backward_D_image()
        self.optimizer_D_image.step()
        
        self.forward()
        self.optimizer_D_person.zero_grad()
        self.backward_D_person()
        self.optimizer_D_person.step()
        
        if only_d == False:
            self.forward()
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()

        self.netD_person.feature_network.requires_grad_(False)

    def get_current_errors(self):
        return OrderedDict([('G_GAN_image', self.loss_G_GAN_image.cpu().data),
                            ('G_GAN_person', self.loss_G_GAN_person.cpu().data),
                            ('G_L1', self.loss_G_L1.cpu().data),
                            #('G_L1_person', self.loss_G_L1_person.data[0]),
                            ('D_image_real', self.loss_D_image_real.cpu().data),
                            ('D_image_fake', self.loss_D_image_fake.cpu().data),
                            ('D_person_real', self.loss_D_person_real.cpu().data),
                            ('D_person_fake', self.loss_D_person_fake.cpu().data),
                            ('G_Crop_Consistency', self.loss_G_Crop_Consistency.cpu().data.numpy())  
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        D2_fake = util.tensor2im(self.person_crop_fake.data)
        D2_real = util.tensor2im(self.person_crop_real.data)

        #print(f"real_A shape: {real_A.shape}")  # Debugging
        #print(f"D2_fake shape: {D2_fake.shape}")  # Debugging

        visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B), ('D2_fake', D2_fake), ('D2_real', D2_real)])

        for i in range(len(self.bbox)):
            if len(self.bbox[i]) == 4:
                y, x, w, h = self.bbox[i]
                display = deepcopy(real_A)
                if len(display.shape) == 3 and len(D2_fake[i].shape) == 3:
                   display[y:h, x:w, :] = D2_fake
                   visuals[f'display_{i}'] = display
                #else:
                   #print(f"Display or D2_fake image {i} does not have the expected 3 dimensions")
            else:
                   print(f"Warning - Incorrect bbox format for image index {i}: {self.bbox[i]}")

        return visuals


    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD_image, 'D_image', label, self.gpu_ids)
        self.save_network(self.netD_person, 'D_person', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D_image.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_person.param_groups:
            param_group['lr'] = lr
                # Set the fixed learning rate for optimizer_G
        fixed_lr = 0.0002
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = fixed_lr
        #for param_group in self.optimizer_G.param_groups:
            #param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
