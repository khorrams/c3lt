import os
import time
import pprint
import glob
import torch
from PIL import Image
from torchvision.utils import save_image
from c3lt.modules import *


def load_pretrained_gan(args):
    """
        loads generator and discriminator from a pretrained gan.
    :param args:
    :return:
    """
    from models import Generator, Discriminator

    generator = Generator(args.latent_dim).to(args.device)
    generator.load_state_dict(torch.load(args.gen_path, map_location=args.device))
    generator.requires_grad = False
    generator.eval()
    print("Generator loaded!")

    # discriminator_enc = Discriminator().to(args.device)
    discriminator = Discriminator().to(args.device)
    discriminator.load_state_dict(torch.load(args.disc_path, map_location=args.device))
    discriminator.requires_grad = False
    discriminator.eval()
    print("Discriminator loaded!")

    return generator, discriminator


def load_pretrained_encoder(args):
    """
        loads a pretrained encoder to get to the latent code given an input image.
    :param args:
    :return:
    """
    from models import EncoderDCGAN

    if args.cls_1 < args.cls_2:
        cls_pair = f"{args.cls_1}_{args.cls_2}"
    else:
        cls_pair = f"{args.cls_2}_{args.cls_1}"

    enc_path = f"models/encoders/encoder_{args.dataset}_{cls_pair}.pt"

    encoder = EncoderDCGAN(args.latent_dim).to(args.device)

    if os.path.exists(enc_path):
        encoder.load_state_dict(torch.load(enc_path))
        encoder.requires_grad = False
        encoder.eval()
        print("Encoder loaded!")
    else:
       raise FileNotFoundError

    return encoder


def load_pretrained_classifier(args):
    """
        loads a pretrained classifier (to be explained).
    :param args:
    :return:
    """
    from models import MnistNet

    model = MnistNet().to(args.device)
    model.load_state_dict(torch.load(args.cls_path, map_location=args.device))
    model.requires_grad = False
    model.eval()
    print("Classifier loaded!")

    return model


def assert_batch_size_equal(batch_1, batch_2):
    """
        sets the batch size to the minimum of the two batches if not equal.
    :param batch_1:
    :param batch_2:
    :return:
    """

    if batch_1.size(0) == batch_2.size(0):
        pass

    elif batch_1.size(0) < batch_2.size(0):
        batch_2 = batch_2[:batch_1.size(0)]

    elif batch_2.size(0) < batch_1.size(0):
        batch_1 = batch_1[:batch_2.size(0)]

    return batch_1, batch_2


def forward_map(imgs, encoder, generator, map_func, args):
    """
        performs an n-step mapping in the latent space.
    :param imgs:
    :param encoder:
    :param generator:
    :param map_func:
    :param args:
    :return:
    """
    z = encoder(imgs)
    out_z = nonlinear_map_step(z, map_func, args.n_steps)
    path_imgs = list(map(generator, out_z))
    return path_imgs, out_z


def nonlinear_map_step(z, map_func, n_steps):
    """
        takes n_steps non-linear mappings staring from z.
    :param z:
    :param map_func:
    :param n_steps:
    :return:

    Inspired from Tensorflow implementation available at https://github.com/ali-design/gan_steerability
    """
    out_to = [z]
    z_shape = z.shape
    z = z.view(z.shape[0], -1)
    z_prev = z
    z_norm = torch.norm(z, dim=1).view(-1, 1)
    for _ in range(1, n_steps + 1):
        z_step = z_prev + map_func(z_prev.view(z_shape)).view(z_shape[0], -1)
        z_step_norm = torch.norm(z_step, dim=1).view(-1, 1)
        z_step = z_step * z_norm / z_step_norm
        out_to.append(z_step.view(z_shape))
        z_prev = z_step
    return out_to


def classifier_loss(classifier, images, target_class, args, reduction='mean'):
    """
        calculates classifier loss in c3lt.
    :param classifier:
    :param images:
    :param target_class:
    :param args:
    :param reduction:
    :return:
    """
    if args.cls_type == "hinge":
        output = torch.nn.Softmax(dim=1)(classifier(images))
        num_samples, num_classes = output.shape
        assert target_class < num_classes

        classes = torch.arange(num_classes)
        non_target_classes = classes[classes != target_class]

        max_output_non_target = torch.amax(output[:, non_target_classes], dim=1)
        output_target = output[:, target_class]
        kappa = torch.empty(num_samples,).fill_(args.kappa).to(images.device)
        loss = torch.maximum(-output_target + max_output_non_target + kappa, torch.zeros(num_samples,).to(images.device))
        return loss.mean() if reduction == "mean" else loss.sum()

    elif args.cls_type == "nll":
        loss = nn.NLLLoss(reduction=reduction)
        target = torch.empty(images.shape[0],).fill_(target_class).to(images.device).long()
        return loss(classifier(images), target)

    else:
        raise ValueError


def proximity_loss(real_images, fake_images, p1=0.0, p2=0.0, reduction='mean'):
    """
        calcualtes proximity loss in c3lt.
    :param real_images:
    :param fake_images:
    :param p1:
    :param p2:
    :param reduction:
    :return:
    """
    masks = gen_masks(real_images, fake_images, mode='mse')
    L1 = nn.L1Loss(reduction=reduction)
    smooth = smoothness_loss(masks, reduction=reduction)
    entropy = entropy_loss(masks, reduction=reduction)
    prx = L1(real_images, fake_images)
    return (prx + p1 * smooth + p2 * entropy) / (1 + p1 + p2)


def smoothness_loss(masks, beta=2, reduction="mean"):
    """
        smoothness loss that encourages smooth masks.
    :param masks:
    :param beta:
    :param reduction:
    :return:
    """
    # TODO RGB images
    masks = masks[:, 0, :, :]
    a = torch.mean(torch.abs((masks[:, :-1, :] - masks[:, 1:, :]).view(masks.shape[0], -1)).pow(beta), dim=1)
    b = torch.mean(torch.abs((masks[:, :, :-1] - masks[:, :, 1:]).view(masks.shape[0], -1)).pow(beta), dim=1)
    if reduction == "mean":
        return (a + b).mean() / 2
    else:
        return (a + b).sum() / 2


def entropy_loss(masks, reduction="mean"):
    """
        entropy loss that encourages binary masks.
    :param masks:
    :param reduction:
    :return:
    """
    # TODO RGB images
    masks = masks[:, 0, :, :]
    b, h, w = masks.shape
    if reduction == "mean":
        return torch.minimum(masks.view(b, -1), 1.0 - masks.view(b, -1)).mean()
    else:
        return torch.minimum(masks.view(b, -1), 1.0 - masks.view(b, -1)).sum()


def perceptual_loss(imgs_1, imgs_2, model, input_dis_penalty=1.0, layers=('layer1', 'layer2'), reduction="mean"):
    """
        calculates perceptual loss given two set of images.
    :param imgs_1:
    :param imgs_2:
    :param model:
    :param input_dis_penalty:
    :param layers:
    :param reduction:
    :return:
    """
    from c3lt.modules import MNISTFeatureExtractor

    featex = MNISTFeatureExtractor(model, layers)
    L1 = torch.nn.L1Loss(reduction=reduction)

    # perceptual loss
    out = 0
    for f, g in zip(featex(imgs_1), featex(imgs_2)):
        out += L1(f, g)
    out += L1(imgs_1, imgs_2) * input_dis_penalty

    return out / (1 + len(layers))


def init_log(args):
    """
        Creates the directories to save the results and args.
    :param args:
    :return:
    """
    # main directories

    main_dir = f"output_{args.method}_{args.dataset}"
    out_dir = os.path.join(main_dir, "results")
    log_dir = os.path.join(main_dir, "logs")

    # start time
    start_time = time.strftime("%m_%d_%Y---%H:%M:%S")

    if args.eval:
        run_name = f'EVAL_{start_time}'
    else:
        run_name = f'{start_time}'

    # create run directories
    out_dir = os.path.join(out_dir, run_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # directory for taking snapshot during training
    if not args.eval:
        snap_dir = os.path.join(out_dir, 'snaps')
        if not os.path.exists(snap_dir):
            os.makedirs(snap_dir)

    # output and log for current run
    args.output = out_dir
    args.log_path = os.path.join(log_dir, run_name)

    # save args into .txt file
    with open(os.path.join(out_dir, 'args.txt'), 'w') as file:
        out = pprint.pformat(args.__dict__)
        file.write(out)


def save_tensors(tensors, args, index="", extra_inf="", preds=None):
    """
        saves tensors into images.
    :param tensors:
    :param args:
    :param index:
    :param extra_inf:
    :return:
    """
    # slice and then save
    tensors = tuple(map(lambda x: x[:args.sample_num], tensors))

    if preds is None:
        save_image(torch.cat(tensors, 0), "{}/{}_{}_{}.png".format(args.output, index, args.method, extra_inf),
                   nrow=args.sample_num, normalize=True, padding=0)
    else:
        # grayscale to rgb
        tensors = tuple(map(lambda x: x.repeat(1, 3, 1, 1), tensors))
        save_image(torch.cat(tensors, 0), "{}/{}_{}_{}.png".format(args.output, index, args.method, extra_inf),
                   nrow=args.sample_num, normalize=True, padding=0)


def save_as_gif(tensors, args, index, folder_name):
    """
        saves tensors into GIF.
    :param tensors:
    :param args:
    :param index:
    :param folder_name:
    :return:
    """
    # slice and then save
    out_dir = os.path.join(args.output, folder_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # save images from tensors
    assert 2*args.sample_num < args.batch_size

    tensors = tuple(map(lambda x: x[:2*args.sample_num], tensors))
    save_image(torch.cat(tensors, 0), "{}/{}.png".format(out_dir, index),
               nrow=2*args.sample_num, normalize=True)

    # file paths
    fp_in = "{}/*.png".format(out_dir)
    fp_out = "{}/{}.gif".format(args.output, folder_name)

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=args.gif_tpf, loop=0)


def gen_masks(inputs, targets, mode='abs'):
    """
        generates a difference masks give two images (inputs and targets).
    :param inputs:
    :param targets:
    :param mode:
    :return:
    """
    # TODO RGB images
    masks = targets - inputs
    masks = masks.view(inputs.size(0), -1)

    if mode == 'abs':
        masks = masks.abs()
        # normalize 0 to 1
        masks -= masks.min(1, keepdim=True)[0]
        masks /= masks.max(1, keepdim=True)[0]

    elif mode == "mse":
        masks = masks ** 2
        masks -= masks.min(1, keepdim=True)[0]
        masks /= masks.max(1, keepdim=True)[0]

    elif mode == 'normal':
        # normalize -1 to 1
        min_m = masks.min(1, keepdim=True)[0]
        max_m = masks.max(1, keepdim=True)[0]
        masks = 2 * (masks - min_m) / (max_m - min_m) - 1

    else:
        raise ValueError("mode value is not valid!")

    return masks.view(inputs.shape)



