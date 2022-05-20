python3 train.py --name train_vox_image+video \
    --image_text_folder data/mmvoxceleb \
    --dataset vox --attr_mode image+video33 --vc_mode face2_8x8 \
    --visual --num_visuals 4 --fullvc --batch_size 32 \
    --text_seq_len 20 \
    --use_html --log_every 200 \
    --sample_every 5000 --n_sample 2 --n_per_sample 4 --num_targets 8 --frame_num 8 \
    --frame_step 4 --image_size 128 \
    --dropout_vc 0.4 --dist_url tcp://localhost:10007 --vae_path pretrained_models/vae_vox.ckpt \
    --cvae_path pretrained_models/cvae_vox.ckpt --rel_no_fully_masked --visual_aug_mode motion_color
