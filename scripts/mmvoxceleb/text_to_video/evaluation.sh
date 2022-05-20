python3 test.py --name test_vox_text \
    --image_text_folder data/mmvoxceleb \
    --dataset video_text --text_seq_len 50 \
    --use_html \
    --num_visuals 0 --num_targets 8 --frame_num 8 \
    --frame_step 4 --image_size 128 \
    --iters 20 --batch_size 16 --n_per_sample 4 \
    --n_sample 1 --no_debug --mp_T 20 --dalle_path vox_bert_text_bs48_100k.pt \
    --eval_mode eval --eval_metric fvd_prd --eval_num 2048 \
    --batch_size 16 --name_suffix _eval=fvd