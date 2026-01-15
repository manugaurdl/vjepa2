from src.datasets.video_dataset import make_videodataset

import os

def get_loaders(args, train_transform, eval_transform, sampling_kwargs, rank, world_size, is_master):
    train_ds, train_loader, train_sampler = make_videodataset(
        data_paths=args.data_path,
        batch_size=args.batch_size,
        frames_per_clip=args.frames_per_clip,
        num_clips=args.num_clips,
        random_clip_sampling=args.random_clip_sampling,
        allow_clip_overlap=args.allow_clip_overlap,
        transform=train_transform,
        shared_transform=None,
        rank=rank,
        world_size=world_size,
        collator=None,
        drop_last=args.drop_last,
        num_workers=args.num_workers,
        pin_mem=args.pin_mem,
        persistent_workers=args.persistent_workers,
        deterministic=args.deterministic_loader,
        log_dir=(os.path.join(args.output_dir, "dataloader_logs") if is_master else None),
        debug=args.debug,
        uniform_sampling=args.uniform_sampling,
        **sampling_kwargs,
    )

    val_ds, val_loader, val_sampler = make_videodataset(
        data_paths=args.val_data_path,
        batch_size=(args.val_batch_size or args.batch_size),
        frames_per_clip=args.eval_frames_per_clip,
        num_clips=args.eval_num_clips,
        random_clip_sampling=False,
        allow_clip_overlap=args.allow_clip_overlap,
        transform=eval_transform,
        shared_transform=None,
        rank=rank,
        world_size=world_size,
        collator=None,
        drop_last=False,
        num_workers=args.num_workers,
        pin_mem=args.pin_mem,
        persistent_workers=args.persistent_workers,
        deterministic=True,
        log_dir=None,
        debug=args.debug,
        uniform_sampling=args.uniform_sampling,
        **sampling_kwargs,
    )

    return train_ds, train_loader, train_sampler, val_ds, val_loader, val_sampler