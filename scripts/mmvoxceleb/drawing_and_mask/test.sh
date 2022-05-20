python3 test.py --name test_vox_draw+mask \
    --image_text_folder data/mmvoxceleb \
    --dataset vox --attr_mode draw+mask2 --visual \
    --vc_mode mask2_8x8 --num_visuals 2 --fullvc \
    --text_seq_len 20 \
    --use_html \
    --num_targets 8 --frame_num 8 --frame_step 4 --image_size 128 \
    --use_cvae --iters 20 --batch_size 16 --n_per_sample 4 --n_sample 1 \
    --no_debug --mp_T 20 --dalle_path vox_bert_draw+mask_bs20_92k.pt
