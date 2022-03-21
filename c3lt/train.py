from c3lt.helper import *
from c3lt.eval import evaluate
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from itertools import chain


def train_c3lt(args, dataloaders):
    """
        trains latent transformations of c3lt.
    :param args:
    :param dataloaders:
    :return:
    """

    writer = SummaryWriter(args.log_path)
    dataloader, dataloader_test = dataloaders
    Tensor = torch.cuda.FloatTensor if args.device == "cuda" else torch.FloatTensor

    # load pretrained modules
    generator, discriminator = load_pretrained_gan(args)
    encoder = load_pretrained_encoder(args)
    classifier = load_pretrained_classifier(args)

    # initialize mapping functions
    g = NLMappingConv(args.latent_dim).to(args.device)  # map c to c'
    h = NLMappingConv(args.latent_dim).to(args.device)  # map c' to c

    # loss functions
    adversarial_loss = torch.nn.BCELoss().to(args.device)
    L1 = torch.nn.L1Loss().to(args.device)

    # init optimizer and scheduler
    optimizer = torch.optim.Adam(
        chain(g.parameters(), h.parameters()),
        lr=args.lr_map
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.decay_step,
        gamma=args.decay_gamma
    )

    # TODO
    # optimizer_D = torch.optim.Adam(chain(discriminator_1.parameters(), discriminator_2.parameters()),
    #                                lr=args.lr_gan,
    #                                betas=(args.b1, args.b2))
    print("\nTraining Starts...\n")
    step = 0
    for epoch in range(0, args.epochs+1):

        total_loss, total_samples = 0, 0
        for i, ((imgs_1, _), (imgs_2, _)) in enumerate(zip(*dataloader)):

            imgs_1 = Variable(imgs_1.type(Tensor), requires_grad=False)
            imgs_2 = Variable(imgs_2.type(Tensor), requires_grad=False)
            imgs_1, imgs_2 = assert_batch_size_equal(imgs_1, imgs_2)
            cur_btch = imgs_1.shape[0]

            # set up optimizer gradient to zero
            optimizer.zero_grad()

            # get latent code from input images
            path_imgs_1, latents_1 = forward_map(imgs_1, encoder, generator, g, args)
            imgs_1_cf, recon_imgs_1 = path_imgs_1[-1], path_imgs_1[0]

            path_imgs_2, latents_2 = forward_map(imgs_2, encoder, generator, h, args)
            imgs_2_cf, recon_imgs_2 = path_imgs_2[-1], path_imgs_2[0]

            # classification loss
            cls_loss = (classifier_loss(classifier, imgs_1_cf, args.cls_2, args) +
                        classifier_loss(classifier, imgs_2_cf, args.cls_1, args)
                        ) / 2

            # proximity loss
            if args.alpha:
                # TODO imgs v.s. recon_imgs
                prox_loss = (proximity_loss(imgs_1, imgs_1_cf, args.p1, args.p2) +
                             proximity_loss(imgs_2, imgs_2_cf, args.p1, args.p2)
                             ) / 2
            else:
                prox_loss = torch.zeros(1).to(args.device)
            
            if args.beta or args.gamma:
                z_1_cyc = nonlinear_map_step(latents_1[-1], h, args.n_steps)[-1]
                z_2_cyc = nonlinear_map_step(latents_2[-1], g, args.n_steps)[-1]
                imgs_1_cyc, imgs_2_cyc = generator(z_1_cyc), generator(z_2_cyc)

            # consistency loss
            if args.beta:
                cyc_loss = (perceptual_loss(imgs_1, imgs_1_cyc, classifier) +
                            perceptual_loss(imgs_2, imgs_2_cyc, classifier) +
                            L1(latents_1[0], z_1_cyc) +
                            L1(latents_2[0], z_2_cyc)
                            ) / 4
            else:
                cyc_loss = torch.zeros(1).to(args.device)

            if args.gamma:
                valid = Variable(torch.empty(imgs_1.shape[0]).fill_(1.0).type(Tensor), requires_grad=False)
                adv_loss = (adversarial_loss(discriminator(imgs_2_cf), valid) +
                            adversarial_loss(discriminator(imgs_1_cf), valid) +
                            adversarial_loss(discriminator(imgs_1_cyc), valid) +
                            adversarial_loss(discriminator(imgs_2_cyc), valid)) / 4
            else:
                adv_loss = torch.zeros(1).to(args.device)

            loss = cls_loss + \
                   args.alpha * prox_loss + \
                   args.beta * cyc_loss + \
                   args.gamma * adv_loss
            loss.backward(retain_graph=False)
            optimizer.step()

            total_loss += loss.item() * cur_btch
            total_samples += cur_btch

            if args.verbose:
                print(f"[Epoch {epoch}/{args.epochs}] "
                      f"[Batch {i}/{len(dataloader[0])}] "
                      f"[CLS loss: {cls_loss.item():.4f}] "
                      f"[PROX loss: {prox_loss.item():.4f}]"
                      f"[CYC loss: {cyc_loss.item():.4f}]"
                      f"[ADV loss: {adv_loss.item():.4f}]"
                      f"[TOTAL loss: {loss.item():.4f}]"
                      )
            writer.add_scalar('c3lt_train/cls', cls_loss.item(), step)
            writer.add_scalar('c3lt_train/prox', prox_loss.item(), step)
            writer.add_scalar('c3lt_train/cyc', cyc_loss.item(), step)
            writer.add_scalar('c3lt_train/adv', adv_loss.item(), step)

            step += 1

            if step % args.sample_interval == 0:
                mask_1 = gen_masks(imgs_1, imgs_1_cf, mode='abs')
                mask_2 = gen_masks(imgs_2, imgs_2_cf, mode='abs')

                if not (args.beta or args.gamma):
                    z_qr_cyc = nonlinear_map_step(latents_1[-1], h, args.n_steps)[-1]
                    z_cf_cyc = nonlinear_map_step(latents_2[-1], g, args.n_steps)[-1]
                    imgs_1_cyc = generator(z_qr_cyc)
                    imgs_2_cyc = generator(z_cf_cyc)

                save_tensors((imgs_1_cyc, recon_imgs_1, imgs_1, mask_1, *path_imgs_1[1:]), args, index=f"TRAIN_Q2C_{step}", extra_inf="")
                save_tensors((imgs_2_cyc, recon_imgs_2, imgs_2, mask_2, *path_imgs_2[1:]), args, index=f"TRAIN_C2Q_{step}", extra_inf="")

        print(f"[Epoch {epoch}/{args.epochs}] [TRAIN Loss: {total_loss/total_samples:.5f}]")

        # evaluate model
        if epoch % args.eval_interval == 0:
            evaluate(encoder, (g, h), generator, discriminator, classifier, dataloader_test, args, writer, epoch)

        scheduler.step()

        writer.add_scalar('c3lt_train/lr', optimizer.param_groups[0]["lr"], epoch)

        if not args.debug and epoch % args.snap_interval == 0:
            torch.save(g.state_dict(), f"{args.output}/snaps/map_{args.cls_1}_to_{args.cls_2}_{epoch}.pt")
            torch.save(h.state_dict(), f"{args.output}/snaps/map_{args.cls_2}_to_{args.cls_1}_{epoch}.pt")

    writer.flush()

    print("\nEvaluation Starts...\n")
    evaluate(encoder, (g, h), generator, discriminator, classifier, dataloader_test, args, writer=None, epoch=-1)


