import argparse
import numpy as np


def get_args_base():
    parser = argparse.ArgumentParser()

    parser.add_argument('--vae_path',
                        type=str,
                        help='path to the pretrained VQGAN for video frames')
    parser.add_argument('--cvae_path',
                        type=str,
                        help='path to the VQGAN for visual controls')
    parser.add_argument(
        '--dalle_path',
        type=str,
        default=None,
        help=
        'path to your mmvid model, used for testing or initializing training')
    parser.add_argument('--which_vae', type=str, default='vqgan1024')
    parser.add_argument('--transformer_path', type=str, default=None)
    parser.add_argument('--image_text_folder',
                        type=str,
                        required=True,
                        help='path to your dataset folder')
    parser.add_argument('--dataset',
                        type=str,
                        default='video_text',
                        help='dataset name')
    parser.add_argument(
        '--dataset_keys',
        type=str,
        default=None,
        help=
        'a text file specifying a subset of keys to use. (a key is the stem without extension)'
    )
    parser.add_argument(
        '--dataset_cache',
        type=str,
        default=None,
        help='path to a cache file (.pkl) to use for the dataset')
    parser.add_argument('--video_only',
                        action='store_true',
                        help='toggle this to use only video without text')
    parser.add_argument(
        '--truncate_captions',
        dest='truncate_captions',
        action='store_true',
        help=
        'Captions passed in which exceed the max token length will be truncated if this is set.'
    )
    parser.add_argument('--random_resize_crop_lower_ratio',
                        dest='resize_ratio',
                        type=float,
                        default=1,
                        help='Random resized crop lower ratio')
    parser.add_argument('--which_tokenizer',
                        type=str,
                        default='simple',
                        help='(yttm | hug | simple | chinese)')
    parser.add_argument('--bpe_path',
                        type=str,
                        help='path to your BPE json file')
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='(experimental) - Enable DeepSpeed 16 bit precision. Reduces VRAM.'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help=
        'Apex "O1" automatic mixed precision. More stable than 16 bit precision. Can\'t be used in conjunction with deepspeed zero stages 1-3.'
    )
    parser.add_argument(
        '--name',
        default='dalle_train_transformer',
        help=
        'experiment name, training logs are stored in log_dir=log_root/name')
    parser.add_argument('--visual',
                        action='store_true',
                        help='add visual control?')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_html',
                        action='store_true',
                        help='use html output?')
    parser.add_argument("--log_root",
                        type=str,
                        default='logs',
                        help='root directory for logs')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--iters',
                        default=200000,
                        type=int,
                        help='Number of iterations')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--deterministic',
                        action='store_true',
                        help='Deterministic in data loader?')
    parser.add_argument('--frame_num',
                        default=8,
                        type=int,
                        help='Number of frames the data loader loads')
    parser.add_argument('--frame_step',
                        default=4,
                        type=int,
                        help='Step size of the data loader')
    parser.add_argument(
        '--rand_visual',
        action='store_true',
        help=
        'toggle this to randomly erase visual control tokens during training')
    parser.add_argument('--fullvc',
                        action='store_true',
                        help='disable random erasing of visual control tokens')
    parser.add_argument(
        '--negvc',
        action='store_true',
        help=
        'toggle this to use negative visual control from data loader. This is used for REL during training. Default is to swap the control sequence along batch.'
    )
    parser.add_argument(
        '--vc_mode',
        type=str,
        default=None,
        help=
        'specifies various modes for masking out visual control tokens, e.g. partially masked faces etc.'
    )
    parser.add_argument(
        '--attr_mode',
        type=str,
        default='object',
        help=
        'specifies various modes for dataset attributes, e.g. mask+text etc.')
    parser.add_argument('--dropout_vc',
                        type=float,
                        default=0.1,
                        help='prob of visual control to be zeroed')
    parser.add_argument(
        '--mask_predict_steps',
        nargs='+',
        default=[0],
        type=int,
        help='mask predict steps, this will override mp_T if nonzero')
    parser.add_argument(
        '--mask_predict_steps1',
        default=0,
        type=int,
        help=
        'mask predict steps for counterfactual samples (conditioned on the current text and the next visual control)'
    )
    parser.add_argument('--n_sample',
                        default=4,
                        type=int,
                        help='Number of samples to visualize')
    parser.add_argument('--n_per_sample',
                        default=4,
                        type=int,
                        help='Number of images per sample to visualize')
    parser.add_argument('--drop_sentence',
                        action='store_true',
                        help='toggle this to use text dropout during training')
    parser.add_argument(
        '--fixed_language_model',
        type=str,
        default=None,
        help=
        'specify a pretrined language model to use for text augmentation, e.g. roberta-large'
    )

    parser.add_argument('--dim',
                        default=768,
                        type=int,
                        help='Model dimension (of embeddings)')
    parser.add_argument('--text_seq_len',
                        default=50,
                        type=int,
                        help='Text sequence length')
    parser.add_argument('--loss_img_weight',
                        default=7,
                        type=int,
                        help='Image loss weight, only used in ART-V')
    parser.add_argument('--which_transformer',
                        type=str,
                        default='openai_clip_visual',
                        help='which transformer to use')
    parser.add_argument(
        '--image_size',
        default=None,
        type=int,
        help=
        'force to use this size if set to > 0, otherwise use VQGAN\'s default size'
    )
    parser.add_argument('--num_targets',
                        default=1,
                        type=int,
                        help='number of frames to generate')
    parser.add_argument('--num_visuals',
                        default=1,
                        type=int,
                        help='number of frames of visual controls')
    parser.add_argument('--use_separate_visual_emb',
                        action='store_true',
                        help='toggle this to use separate visual embeddings')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument(
        '--text_emb_bottleneck',
        type=str,
        default=None,
        help=
        'if we use a pretrained language model, we can specify the bottleneck layer (number of channels) to use'
    )
    parser.add_argument(
        '--visual_aug_mode',
        type=str,
        default=None,
        help=
        'specify the visual augmentation mode, e.g. motion_color will add color jitter (used when conditioning on image and video)'
    )

    parser.add_argument('--mp_T1n',
                        type=int,
                        default=10,
                        help='L1, number of steps for mask')
    parser.add_argument('--mp_T2n',
                        type=int,
                        default=10,
                        help='L2, number of steps for mask')
    parser.add_argument('--mp_T3n',
                        type=int,
                        default=30,
                        help='L3, number of steps for mask')
    parser.add_argument('--mp_N1n',
                        type=float,
                        default=0.9,
                        help='alpha1 for mask')
    parser.add_argument('--mp_N2n',
                        type=float,
                        default=0.1,
                        help='beta1 for mask')
    parser.add_argument('--mp_N3n',
                        type=float,
                        default=0.125,
                        help='alpha2 for mask')
    parser.add_argument('--mp_N4n',
                        type=float,
                        default=0.0625,
                        help='alpha3 for mask')
    parser.add_argument('--mp_T1t',
                        type=int,
                        default=10,
                        help='L1, number of steps for noise')
    parser.add_argument('--mp_T2t',
                        type=int,
                        default=5,
                        help='L2, number of steps for noise')
    parser.add_argument('--mp_T3t',
                        type=int,
                        default=35,
                        help='L3, number of steps for noise')
    parser.add_argument('--mp_N1t',
                        type=float,
                        default=0.,
                        help='alpha1 for noise')
    parser.add_argument('--mp_N2t',
                        type=float,
                        default=0.,
                        help='beta1 for noise')
    parser.add_argument('--mp_N3t',
                        type=float,
                        default=0.,
                        help='alpha2 for noise')
    parser.add_argument('--mp_N4t',
                        type=float,
                        default=0.,
                        help='alpha3 for noise')
    parser.add_argument('--mp_T',
                        type=int,
                        default=20,
                        help='number of total steps for mask-predict')
    parser.add_argument('--mp_B', type=int, default=1, help='beam search size')

    parser.add_argument(
        '--ar',
        action='store_true',
        help='toggle this to use auto-regressive model (ART-V)')
    parser.add_argument(
        '--slow',
        action='store_true',
        help=
        'toggle this to add speed variants to dataset, used in iPER. Please see iPER dataset for details'
    )
    parser.add_argument(
        '--insert_sep',
        action='store_true',
        help=
        'toggle this to insert a separator token between visual control frames'
    )
    parser.add_argument(
        '--pnag_argmax',
        action='store_true',
        help=
        'toggle this to use argmax in mask predict, inspired by PNAG in UFC-BERT'
    )
    parser.add_argument(
        '--pnag_dynamic',
        action='store_true',
        help=
        'toggle this to use dynamic stopping in mask predict, inspired by PNAG in UFC-BERT'
    )
    parser.add_argument(
        '--openai_clip_model_path',
        type=str,
        default='ViT-B-32.pt',
        help=
        "you can download from here: https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    )
    return parser


def get_args_train():
    parser = get_args_base()
    parser.add_argument('--rank',
                        type=int,
                        default=0,
                        help='node rank for distributed training')
    parser.add_argument('--gpu_ids',
                        type=int,
                        default=None,
                        help='gpu id to use')
    parser.add_argument('--workers',
                        default=16,
                        type=int,
                        help='# data loading workers')
    parser.add_argument('--world_size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist_url',
                        default='tcp://localhost:10001',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend',
                        default='nccl',
                        type=str,
                        help='distributed backend')
    parser.add_argument(
        '--multiprocessing_distributed',
        action='store_true',
        help=
        'Use multi-processing distributed training to launch N processes per node, which has N GPUs.'
    )
    parser.add_argument('--save_every_n_steps',
                        default=5000,
                        type=int,
                        help='Save a checkpoint every n steps')
    parser.add_argument('--learning_rate',
                        default=1e-4,
                        type=float,
                        help='Learning rate')
    parser.add_argument('--clip_grad_norm',
                        default=1.0,
                        type=float,
                        help='Clip gradient norm')
    parser.add_argument('--no_lr_decay',
                        action='store_true',
                        help='toggle this to disable learning rate decay')
    parser.add_argument("--log_every",
                        type=int,
                        default=200,
                        help="logging every # iters")
    parser.add_argument("--sample_every",
                        type=int,
                        default=5000,
                        help="sample every # iters")
    parser.add_argument('--start_iter',
                        default=None,
                        type=int,
                        help='start from this iter')
    parser.add_argument("--limit_train_batches",
                        type=float,
                        default=1,
                        help='similar to pytorch-lightning')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr_scheduler', type=str, default='warmuplr')
    parser.add_argument('--lr_scheduler_every',
                        default=1,
                        type=int,
                        help='step lr scheduler every n steps')
    parser.add_argument('--lr_scheduler_step_size',
                        default=10000,
                        type=int,
                        help='used in steplr and cosineannealinglr')
    parser.add_argument('--lr_scheduler_warmup',
                        default=5000,
                        type=int,
                        help='warmup steps for warmuplr')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--beta_msm',
                        default=7.0,
                        type=float,
                        help='weight for MSM (or MLM) loss')
    parser.add_argument('--beta_rel',
                        default=0.5,
                        type=float,
                        help='weight for REL loss')
    parser.add_argument('--beta_vid',
                        default=0.5,
                        type=float,
                        help='weight for VID loss')
    parser.add_argument(
        '--msm_strategy_prob',
        type=str,
        default='7,1,1,1',
        help=
        'comma separated list, specifies the probabilites of sampling each masking strategy'
    )
    parser.add_argument(
        '--msm_bernoulli_prob',
        type=str,
        default='0.2,0.2',
        help=
        'comma separated list, specifies the min and max probabilities of random Bernoulli masking'
    )
    parser.add_argument('--vid_strategy_prob',
                        type=str,
                        default='1,1,1,1',
                        help='comma separated list')
    parser.add_argument(
        '--rel_no_fully_masked',
        action='store_true',
        help='toggle this to exclude fully masked sequences in REL loss')
    parser.add_argument('--pc_prob',
                        type=float,
                        default=0,
                        help='prob of preservation control')
    args = parser.parse_args()

    return args, parser


def get_args_test():
    parser = get_args_base()
    parser.add_argument(
        '--name_suffix',
        default='',
        type=str,
        help=
        'suffix to add to the experiment name, if you don\'t want to mess up the original log files (in training)'
    )
    parser.add_argument('--test_mode',
                        type=str,
                        default=None,
                        help='used in testing, e.g. ')
    parser.add_argument('--eval_mode', type=str, default=None)
    parser.add_argument('--eval_metric',
                        type=str,
                        nargs='+',
                        default=['fvd_prd'])
    parser.add_argument('--eval_num', type=int, default=2048)
    parser.add_argument(
        '--pc_mode',
        type=str,
        default=None,
        help=
        'reserved for future use (if we want to use more complicated preservation control)'
    )
    parser.add_argument(
        '--description',
        type=str,
        default=None,
        help=
        'specify a text prompt, which will overwrite the text from dataloader')
    parser.add_argument('--no_debug', action='store_true')
    parser.add_argument('--t_overlap',
                        default=1,
                        type=int,
                        help='in long sequence, how many frames to overlap')
    parser.add_argument(
        '--t_repeat',
        default=10,
        type=int,
        help='in long sequence generation mode, repeat sampling this many times'
    )
    parser.add_argument('--use_cvae', action='store_true')
    parser.add_argument('--save_codebook', action='store_true')
    parser.add_argument(
        '--long_mode',
        type=str,
        default='long',
        help=
        'specify long sequence generation mode, e.g. long, interp, interp_real'
    )
    args = parser.parse_args()

    return args, parser


def process_args(train=False):
    if train:
        args, _ = get_args_train()
    else:
        args, _ = get_args_test()
    # Mask-Predict hyperparameters
    mp_config = {
        'T1_n': args.mp_T1n,
        'T2_n': args.mp_T2n,
        'T3_n': args.mp_T3n,
        'N1_n': args.mp_N1n,
        'N2_n': args.mp_N2n,
        'N3_n': args.mp_N3n,
        'N4_n': args.mp_N4n,
        'T1_t': args.mp_T1t,
        'T2_t': args.mp_T2t,
        'T3_t': args.mp_T3t,
        'N1_t': args.mp_N1t,
        'N2_t': args.mp_N2t,
        'N3_t': args.mp_N3t,
        'N4_t': args.mp_N4t,
        'T': args.mp_T,
        'B': args.mp_B,
    }
    args.mp_config = mp_config

    # constants
    args.truncate_captions = True
    args.num_visuals *= args.visual

    if args.ar:
        args.debug = False
        args.mask_predict_steps = [0]
        args.mask_predict_steps1 = 0
        args.num_visuals = max(1, args.num_visuals)

    if train:
        if args.ar:
            args.beta_msm = 1.0
        args.lr_decay = not args.no_lr_decay
        if args.msm_strategy_prob is not None:
            msm_strategy_prob = np.array(
                list(map(float, args.msm_strategy_prob.split(','))))
            msm_strategy_prob /= msm_strategy_prob.sum()
            args.msm_strategy_prob = msm_strategy_prob

        if args.vid_strategy_prob is not None:
            vid_strategy_prob = np.array(
                list(map(float, args.vid_strategy_prob.split(','))))
            vid_strategy_prob /= vid_strategy_prob.sum()
            args.vid_strategy_prob = vid_strategy_prob

        args.msm_bernoulli_prob = list(
            map(float, args.msm_bernoulli_prob.split(',')))

    else:  # test
        # NOTE: vae weights will be loaded from dalle model checkpoint
        args.vae_path = ""
        args.cvae_path = ""  # NOTE: toggle args.use_cvae to use cvae

    return args