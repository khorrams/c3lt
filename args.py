import argparse


def init_parser():
    parser = argparse.ArgumentParser()

    # ----- training args -----
    parser.add_argument("--epochs",
                        type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size",
                        type=int, default=128, help="size of the batches")
    parser.add_argument("--lr",
                        type=float, default=2e-4, help="adam: learning rate")
    parser.add_argument("--lr_map",
                        type=float, default=1e-3, help="learning rate for the mapper!")
    # parser.add_argument("--lr_gan",
    #                     type=float, default=2e-4, help="learning rate for the Gen and Disc!")
    parser.add_argument("--b1",
                        type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2",
                        type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu",
                        type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--seed",
                        type=int, default=85, help="random seed")
    parser.add_argument('--device',
                        type=str, default="cuda", help="device to use: cpu/cuda")
    parser.add_argument("--debug",
                        action='store_true', help="run with saving checkpoints.")
    parser.add_argument("--eval",
                        action='store_true', help="evaluation (no training) with saved checkpoints.")

    # ----- logging args -----
    parser.add_argument("--sample_interval",
                        type=int, default=200, help="interval (#steps) between image sampling")
    parser.add_argument("--eval_interval",
                        type=int, default=10, help="interval (#epochs) between evaluation")
    parser.add_argument("--snap_interval",
                        type=int, default=25, help="interval (#epochs) between snapshot the models.")
    parser.add_argument("--sample_num",
                        type=int, default=10, help="number of samples shown at each interval")
    parser.add_argument("--verbose",
                        action='store_true', help="monitor additional loss values during training")

    # parser.add_argument("--gi f_tpf",
    #                     type=int, default=1500, help="duration of displaying each frame in the GIF")

    # ----- dataset args -----
    parser.add_argument("--dataset",
                        type=str, default="mnist", help="dataset name")
    parser.add_argument("--dataset_path",
                        type=str, default="./data", help="dataset path")
    parser.add_argument("--img_size",
                        type=int, default=28, help="input image size")
    parser.add_argument("--channels",
                        type=int, default=1, help="number of input image channels")

    # ----- CounterFactual Generation args -----
    parser.add_argument("--method",
                        type=str, default="C3LT", help="method name")
    parser.add_argument("--cls_1",
                        type=int, default=4, help="first (input) class index")
    parser.add_argument("--cls_2",
                        type=int, default=9, help="second (CF) class index")

    # C3LT setup
    parser.add_argument("--gen_path",
                        type=str, default="models/gans/netG_epoch_99.pth", help="path to pretrained generator")
    parser.add_argument("--disc_path",
                        type=str, default="models/gans/netD_epoch_99.pth", help="path to pretrained discriminator")
    parser.add_argument("--cls_path",
                        type=str, default="models/classifiers/mnist.pt", help="path to pretrained classifier")
    parser.add_argument("--latent_dim",
                        type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--n_steps",
                        type=int, default=1, help="number of steps for the walk in the latent space")
    parser.add_argument("--alpha",
                        type=float, default=0.1, help="coefficient for classification loss")
    parser.add_argument("--beta",
                        type=float, default=0.1, help="coefficient for consistency loss")
    parser.add_argument("--gamma",
                        type=float, default=0.0, help="coefficient for adversarial loss")
    parser.add_argument("--p1",
                        type=float, default=10.0, help="penalty coefficient for smoothness loss")
    parser.add_argument("--p2",
                        type=float, default=1.0, help="penalty coefficient for entropy loss")
    parser.add_argument("--cls_type",
                        type=str, default="nll", help="classification loss type",
                        choices=["hinge", "nll"])
    parser.add_argument("--kappa",
                        type=float, default=0.05, help="kappa for hinge loss")
    parser.add_argument("--decay_step",
                        type=int, default=25, help="learning rate decay interval (#epochs)")
    parser.add_argument("--decay_gamma",
                        type=float, default=0.50, help="learning rate decay coefficient")  # 0.912  # 0.8709

    args = parser.parse_args()
    return args
