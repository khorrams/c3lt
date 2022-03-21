from c3lt.helper import *
from torch.autograd import Variable


def eval_c3lt(args, dataloaders):
    """
        evaluates a set of pretrained c3lt transformations.
    :param args:
    :param dataloaders:
    :return:
    """

    _, dataloader_test = dataloaders

    # load pretrained modules
    generator, discriminator = load_pretrained_gan(args)
    encoder = load_pretrained_encoder(args)
    classifier = load_pretrained_classifier(args)

    # initialize mapping functions
    g = NLMappingConv(args.latent_dim).to(args.device)  # map c to c'
    h = NLMappingConv(args.latent_dim).to(args.device)  # map c' to c

    tag = f"snaps/c3lt/{args.dataset}"
    args.map_to_path = f"{tag}/map_{args.query_cls}_to_{args.cf_cls}_250.pt"
    args.h_path = f"{tag}/map_{args.cf_cls}_to_{args.query_cls}_250.pt"
    
    # load mappings
    g.load_state_dict(torch.load(args.map_to_path))
    h.load_state_dict(torch.load(args.h_path))

    evaluate(encoder, (g, h), generator, discriminator, classifier, dataloader_test, args, writer=None, epoch=-1)


def evaluate(encoder, maps, generator, discriminator, classifier, dataloader, args, writer, epoch=-1):
    """
        evaluates loss values and metrics.
    :param encoder:
    :param maps:
    :param generator:
    :param discriminator:
    :param classifier:
    :param dataloader:
    :param args:
    :param writer:
    :param epoch:
    :return:
    """
    # eval params
    cout_num_steps = 50

    # tensor type
    Tensor = torch.cuda.FloatTensor if args.device == "cuda" else torch.FloatTensor

    # get the mappings
    g, h = maps
    g.eval()
    h.eval()

    if epoch == -1:
        fake_img_path_q = os.path.join(args.output, "eval_imgs_1_cf")
        fake_img_path_c = os.path.join(args.output, "eval_imgs_2_cf")
    else:
        fake_img_path_q = os.path.join(args.output, ".temp_1")
        fake_img_path_c = os.path.join(args.output, ".temp_2")

    # save eval images
    os.makedirs(os.path.join(args.output, "EVAL_Q2C"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "EVAL_C2Q"), exist_ok=True)

    # save (temps) cfs
    os.makedirs(fake_img_path_q, exist_ok=True)
    os.makedirs(fake_img_path_c, exist_ok=True)

    # loss functions
    adversarial_loss = torch.nn.BCELoss().to(args.device)
    L1 = torch.nn.L1Loss().to(args.device)

    # init scores
    prox_loss, cls_loss, adv_loss, cyc_loss = 0, 0, 0, 0
    prox, val, cout = 0, 0, 0
    total_samples = 0

    with torch.no_grad():
        for i, ((imgs_1, _), (imgs_2, _)) in enumerate(zip(*dataloader)):
            imgs_1 = Variable(imgs_1.type(Tensor), requires_grad=False)
            imgs_2 = Variable(imgs_2.type(Tensor), requires_grad=False)
            imgs_1, imgs_2 = assert_batch_size_equal(imgs_1, imgs_2)
            cur_btch = imgs_1.shape[0]

            # get latent code from input images
            path_imgs_1, latents_1 = forward_map(imgs_1, encoder, generator, g, args)
            imgs_1_cf, recon_imgs_1 = path_imgs_1[-1], path_imgs_1[0]

            path_imgs_2, latents_2 = forward_map(imgs_2, encoder, generator, h, args)
            imgs_2_cf, recon_imgs_2 = path_imgs_2[-1], path_imgs_2[0]

            # classification loss
            cls_loss += (classifier_loss(classifier, imgs_1_cf, args.cls_2, args) +
                         classifier_loss(classifier, imgs_2_cf, args.cls_1, args)
                         ) / 2 * cur_btch

            prox_loss += (proximity_loss(imgs_1, imgs_1_cf, args.p1, args.p2) +
                          proximity_loss(imgs_2, imgs_2_cf, args.p1, args.p2)
                          ) / 2 * cur_btch

            # consistency loss
            z_1_cyc = nonlinear_map_step(latents_1[-1], h, args.n_steps)[-1]
            z_2_cyc = nonlinear_map_step(latents_2[-1], g, args.n_steps)[-1]
            imgs_1_cyc, imgs_2_cyc = generator(z_1_cyc), generator(z_2_cyc)

            cyc_loss += (perceptual_loss(imgs_1, imgs_1_cyc, classifier) +
                         perceptual_loss(imgs_2, imgs_2_cyc, classifier) +
                         L1(latents_1[0], z_1_cyc) +
                         L1(latents_2[0], z_2_cyc)
                         ) / 4 * cur_btch

            valid = Variable(torch.empty(imgs_1.shape[0]).fill_(1.0).type(Tensor), requires_grad=False)
            adv_loss += (adversarial_loss(discriminator(imgs_2_cf), valid) +
                         adversarial_loss(discriminator(imgs_1_cf), valid) +
                         adversarial_loss(discriminator(imgs_1_cyc), valid) +
                         adversarial_loss(discriminator(imgs_2_cyc), valid)) / 4 * cur_btch
            
            # fix images
            if i == 0 and epoch != -1:
                save_tensors((imgs_1, *path_imgs_1[1:]), args, f"EVAL_Q2C/Epoch_{epoch}", "")
                save_tensors((imgs_2, *path_imgs_2[1:]), args, f"EVAL_C2Q/Epoch_{epoch}", "")

            # calculate metrics
            prox += calculate_proximity(imgs_1, imgs_1_cf)
            val += calculate_validity(classifier, imgs_1_cf, args.cls_2)
            cout_score, _ = calculate_cout(
                imgs_1,
                imgs_1_cf,
                gen_masks(imgs_1, imgs_1_cf, mode='abs'),
                classifier,
                args.cls_1,
                args.cls_2,
                max(1, args.img_size ** 2 // cout_num_steps),
            )
            cout += cout_score

            # update total sample number
            total_samples += cur_btch

    #
    cls_loss /= total_samples
    prox_loss /= total_samples
    cyc_loss /= total_samples
    adv_loss /= total_samples

    prox /= (total_samples * args.img_size ** 2)
    val /= total_samples
    cout /= total_samples

    # total loss calculation
    total_loss = cls_loss + args.alpha * prox_loss + args.beta * cyc_loss + args.gamma * adv_loss

    # print out metrics and loss values
    loss_values = f"\nEVAL [CLS loss: {cls_loss:.4f}]  [PROX loss: {prox_loss:.4f}]  " \
        f"[CYC Loss: {cyc_loss:.4f}]  [ADV loss: {adv_loss:.4f}]  [EVAL loss: {total_loss:.4f}]"
    metrics = f"\nEVAL [COUT: {cout:.4f}]  [Validity: {val*100:.2f} %]  [Proximity: {prox:.4f}]\n"
    print(loss_values, metrics)

    if writer:
        writer.add_scalar('c3lt_eval/cls', cls_loss, epoch)
        writer.add_scalar('c3lt_eval/prox', prox_loss, epoch)
        writer.add_scalar('c3lt_eval/cyc', cyc_loss, epoch)
        writer.add_scalar('c3lt_eval/adv', adv_loss, epoch)
        writer.add_scalar('c3lt_eval/total', total_loss, epoch)
        writer.add_scalar('c3lt_eval/cout', cout, epoch)
        writer.add_scalar('c3lt_eval/val', val, epoch)
        writer.add_scalar('c3lt_eval/prx', prox, epoch)

    # train mode
    g.train()
    h.train()


def calculate_proximity(imgs, imgs_cf):
    """
        calculates the proximity score.
    :param imgs:
    :param imgs_cf:
    :return:
    """
    return abs(imgs - imgs_cf).sum().item()


def calculate_validity(classifier, imgs, target_cls):
    """
        calcuates the validity score.
    :param classifier:
    :param imgs:
    :param target_cls:
    :return:
    """
    with torch.no_grad():
        target = torch.empty(imgs.shape[0]).fill_(target_cls).long().cuda()
        output = classifier(imgs)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        validity = pred.eq(target.view_as(pred)).sum().item()
        return validity


def calculate_cout(imgs, cfs, masks, model, cls_1, cls_2, step):
    """
        calculates the counterfactual transition (cout) score.
    :param imgs:
    :param cfs:
    :param masks:
    :param model:
    :param cls_1:
    :param cls_2:
    :param step:
    :return:
    """
    with torch.no_grad():
        # The dimensions for the image
        img_size = imgs.shape[-1]
        mask_size = masks.shape[-1]

        # Compute the total number of pixels in a mask
        num_pixels = torch.prod(torch.tensor(masks.shape[1:])).item()
        l = torch.arange(imgs.shape[0])

        # Initial values for the curves
        output = torch.nn.Softmax(dim=1)(model(imgs))
        c_curve = [output[:, cls_1]]
        c_prime_curve = [output[:, cls_2]]
        index = [0.]

        # init upsampler
        up_sample = torch.nn.UpsamplingBilinear2d(size=(img_size, img_size)).to(imgs.device)

        # updating mask and the ordering
        cur_mask = torch.zeros((masks.shape[0], num_pixels)).to(imgs.device)
        elements = torch.argsort(masks.view(masks.shape[0], -1), dim=1, descending=True)

        for pixels in range(0, num_pixels, step):
            # Get the indices used in this iteration
            indices = elements[l, pixels:pixels + step].squeeze().view(imgs.shape[0], -1)

            # Set those indices to 1
            cur_mask[l, indices.permute(1, 0)] = 1
            up_masks = up_sample(cur_mask.view(-1, 1, mask_size, mask_size))

            # perturb the image using cur mask and calculate scores
            perturbed = phi(cfs, imgs, up_masks)
            outputs = torch.nn.Softmax(dim=1)(model(perturbed))

            # obtain the scores
            c_curve.append(outputs[:, cls_1])
            c_prime_curve.append(outputs[:, cls_2])
            index.append((pixels + step) / num_pixels)

        auc_c, auc_c_prime = auc(c_curve), auc(c_prime_curve)
        auc_c *= step / mask_size ** 2
        auc_c_prime *= step / mask_size ** 2
        cout = auc_c_prime.sum().item() - auc_c.sum().item()

    return cout, (c_curve, c_prime_curve, index)


def phi(img, baseline, mask):
    """
        composes an image from img and baseline according to the mask values.
    :param img:
    :param baseline:
    :param mask:
    :return:
    """
    return img.mul(mask) + baseline.mul(1-mask)


def auc(curve):
    """
        calculates the area under the curve
    :param curve:
    :return:
    """
    return curve[0]/2 + sum(curve[1:-1]) + curve[-1]/2

