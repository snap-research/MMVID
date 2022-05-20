from pathlib import Path
import os
import random

from datetime import datetime
from tqdm import tqdm

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist

from utils import utils_html
from utils.utils_train import get_dataset, get_fixed_language_model, \
    save_model, prepare_lr_scheduler, dummy_lr_scheduler_step, \
        get_optimizer, get_vae_model, get_tokenizer
from utils.utils_train import visualize_train as visualize
from utils.utils import seed_everything, sample_data, exists, mean_pooling


def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]


def model_to_gpu(model, gpu, is_train):
    if is_train:
        if gpu is not None:
            model.cuda(gpu)
            model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = DDP(model, find_unused_parameters=True)
    else:
        model.cuda()
        model = nn.DataParallel(model)

    return model


def cleanup():
    dist.destroy_process_group()


def main():

    from utils.utils_args import process_args
    args = process_args(train=True)

    args.multiprocessing_distributed = True  # always use multiprocessing_distributed
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.world_batch_size = args.batch_size
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker,
                 nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        main_worker(
            args.gpu_ids,
            ngpus_per_node,
            args,
        )


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    torch.backends.cudnn.benchmark = True

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.multiprocessing_distributed and args.gpu != 0:

        def print_pass(*args):
            pass

        __builtins__['print'] = print_pass

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + args.gpu

        seed_everything(args.seed + args.rank)

        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)

    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    assert Path(args.image_text_folder).exists(
    ), f'The path {args.image_text_folder} was not found.'

    def is_root_worker():
        return (not args.multiprocessing_distributed
                or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0))

    # logging
    log_dir = Path(args.log_root) / args.name
    log_file_name = log_dir / 'log.txt'
    args.log_dir = log_dir
    args.log_sample_dir = log_dir / 'samples'
    ckpt_dir = log_dir / 'weights'

    if is_root_worker():
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(log_dir / 'samples', exist_ok=True)
        os.makedirs(log_dir / 'weights', exist_ok=True)

    webpage = None
    if args.use_html and is_root_worker():
        webpage = utils_html.initialize_webpage(log_dir / 'web',
                                                'DALLE: ' + args.name, False)

    # tokenizer
    if args.fixed_language_model is not None:
        tokenizer2, language_model, text_feature_dim, _ = get_fixed_language_model(args)
        language_model = model_to_gpu(language_model, args.gpu, True)
        tokenizer = None  # TODO: avoid tokenization and get raw text
    else:
        text_feature_dim = 0
        tokenizer = get_tokenizer(args)

    # model path
    args.use_cvae = args.cvae_path is not None
    dalle_path = Path(args.dalle_path) if exists(args.dalle_path) else None

    model_weights, optim_weights = None, None
    start_iter = args.start_iter or 0

    # get vae model
    vae, vae_params = get_vae_model(
        args.which_vae,
        vae_path=args.vae_path,
        image_size=args.image_size,
        args=args,
    )

    cvae = None
    if args.cvae_path is not None:
        cvae, _ = get_vae_model(
            args.which_vae,
            vae_path=args.cvae_path,
            image_size=args.image_size,
            args=args,
        )

    dalle_params = dict(
        num_text_tokens=tokenizer.vocab_size if tokenizer else 0,
        text_seq_len=args.text_seq_len,
        dim=args.dim,
        loss_img_weight=args.loss_img_weight,
        text_feature_dim=text_feature_dim,
        fixed_language_model=args.fixed_language_model,
        text_emb_bottleneck=args.text_emb_bottleneck,
        which_transformer=args.which_transformer,
        num_targets=args.num_targets,
        num_visuals=args.num_visuals,
        use_separate_visual_emb=args.use_separate_visual_emb,
        insert_sep=args.insert_sep,
        openai_clip_path=args.openai_clip_model_path,
    )

    # if not resume and dalle_path is given, load weights
    if dalle_path is not None:
        assert exists(dalle_path), 'DALLE model file does not exist'
        ckpt = torch.load(str(dalle_path))
        model_weights = ckpt['weights']

    image_size = args.image_size or vae.image_size
    args.image_size = vae.image_size = image_size
    if cvae is not None:
        cvae.image_size = image_size

    # initialize DALL-E / BERT and optimizer
    if args.ar:
        from mmvid_pytorch.dalle_artv import DALLE
        dalle = DALLE(vae=vae, cvae=cvae, **dalle_params)
    else:
        from mmvid_pytorch.dalle_bert import BERT
        dalle = BERT(vae=vae, cvae=cvae, **dalle_params)
    if args.fp16:
        dalle = dalle.half()

    opt = get_optimizer(args, get_trainable_params(dalle))

    if model_weights is not None:
        dalle.load_state_dict(model_weights)
        del ckpt
    if optim_weights is not None:
        opt.load_state_dict(optim_weights)

    dalle = model_to_gpu(dalle, args.gpu, True)
    dalle_module = dalle.module

    scheduler_step = dummy_lr_scheduler_step
    if args.lr_decay:
        _, scheduler_step = prepare_lr_scheduler(args, opt)

    # create dataset and dataloader
    args.is_shuffle = True

    ds = get_dataset(args, tokenizer)
    assert len(ds) > 0, 'dataset is empty'
    if args.limit_train_batches < 1:
        indices = torch.randperm(len(ds))[:int(args.limit_train_batches *
                                               len(ds))]
        ds = torch.utils.data.Subset(ds, indices)
    if is_root_worker():
        print(f'{len(ds)} image-text pairs found for training')

    data_sampler = torch.utils.data.distributed.DistributedSampler(ds)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(data_sampler is None),
        drop_last=True,
        sampler=data_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if is_root_worker():
        with open(log_file_name, 'a+') as f:
            f.write(
                f"Name: {getattr(args, 'name', 'NA')} Time: {datetime.now()}\n{'-'*50}\n"
            )

    distr_dl_iter = sample_data(dl, data_sampler)

    if is_root_worker():
        pbar = tqdm(range(args.iters),
                    initial=start_iter,
                    dynamic_ncols=True,
                    smoothing=0.01)

    # training
    for idx in pbar:
        i = idx + start_iter
        which_iter = f"{i:07d}"

        if i > args.iters:
            print('done!')
            break

        if args.negvc:
            text, frames, visuals, visuals_neg, text_neg = next(distr_dl_iter)
            visuals_neg, text_neg = map(lambda t: t.cuda(),
                                        (visuals_neg, text_neg))
        else:
            # frames [B, T, C, H, W]
            text, frames, visuals = next(distr_dl_iter)
        if args.visual and len(visuals.shape) == 4:
            assert args.num_visuals == 1
            visuals = visuals.unsqueeze(1)

        if args.fp16:
            frames = frames.half()
        frames, visuals = map(lambda t: t.cuda(), (frames, visuals))

        if args.fixed_language_model is not None:
            text_description = text
            with torch.no_grad():
                encoded_input = tokenizer2(
                    text_description,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=args.text_seq_len,
                )
                encoded_input = {
                    'input_ids': encoded_input['input_ids'].cuda(),
                    'attention_mask': encoded_input['attention_mask'].cuda(),
                }
                model_output = language_model(**encoded_input)
                text = mean_pooling(model_output,
                                    encoded_input['attention_mask'])
        else:
            text = text.cuda()
            text_description = None

        target = frames[:, :args.num_targets, ...]

        # Train dalle
        loss_msm, loss_rel, loss_vid = dalle(
            text,
            visual=visuals if
            (args.visual and
             (args.fullvc or random.random() >= args.dropout_vc)) else None,
            target=target,
            erase_visual=args.rand_visual,
            return_loss=True,
            rel=args.beta_rel > 0,
            vid=args.beta_vid > 0,
            msm_strategy_prob=args.msm_strategy_prob,
            msm_bernoulli_prob=args.msm_bernoulli_prob,
            vid_strategy_prob=args.vid_strategy_prob,
            rel_no_fully_masked=args.rel_no_fully_masked,
            negvc=args.negvc,
            visual_neg=visuals_neg if (args.negvc and args.visual) else None,
            text_neg=text_neg if args.negvc else None,
            pc_prob=args.pc_prob,
            vc_mode=args.vc_mode,
            face_mode=None,  # NOTE: face_mode is used in testing, for specifying the desired occlusion pattern
            visual_aug_mode=args.visual_aug_mode,
        )
        loss = args.beta_msm * loss_msm + args.beta_rel * loss_rel + args.beta_vid * loss_vid

        opt.zero_grad()
        loss.backward()
        clip_grad_norm_(dalle.parameters(), args.clip_grad_norm)
        opt.step()

        avg_loss = loss

        if is_root_worker():
            pbar.set_description((f"loss {avg_loss.item():.4f} "))

        if i % args.log_every == 0 and is_root_worker():
            with open(log_file_name, 'a+') as f:
                f.write((f"iter {i:07d}; "
                         f"MSM {loss_msm.item():.4f}; "
                         f"REL {loss_rel.item():.4f}; "
                         f"VID {loss_vid.item():.4f}; "
                         f"lr {opt.param_groups[0]['lr']}"
                         f"\n"))

        if args.save_every_n_steps > 0 and i % args.save_every_n_steps == 0 and is_root_worker(
        ):
            save_model(
                ckpt_dir / which_iter,
                params={
                    'iter': i,
                    'hparams': dalle_params,
                    'vae_params': vae_params,
                },
                states={
                    'weights': dalle_module.state_dict(),
                    'optimizer': opt.state_dict(),
                },
            )

        # =================== visualization ======================
        if i % args.sample_every == 0 and is_root_worker():
            visualize(
                args,
                dalle_module,
                tokenizer,
                {
                    'description': text_description,
                    'text': text,
                    'target': frames,
                    'visual': visuals,
                },
                which_iter,
                webpage,
            )
        # ========================================================

        if args.lr_decay and (i + 1) % args.lr_scheduler_every == 0:
            scheduler_step(avg_loss)

    # finish
    if is_root_worker():
        save_model(
            ckpt_dir / 'last',
            params={
                'iter': i,
                'hparams': dalle_params,
                'vae_params': vae_params,
            },
            states={
                'weights': dalle_module.state_dict(),
                'optimizer': opt.state_dict(),
            },
        )

    cleanup()


if __name__ == "__main__":
    main()
