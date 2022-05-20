python3 test.py --name test_vox_text+draw \
    --image_text_folder data/mmvoxceleb \
    --dataset vox --attr_mode draw+text_dropout \
    --visual --vc_mode mask_8x8 --num_visuals 1 --fullvc \
    --text_seq_len 50 \
    --use_html --num_targets 8 --frame_num 8 --frame_step 4 \
    --image_size 128 --use_cvae --iters 20 \
    --batch_size 16 --n_per_sample 4 --n_sample 1 --no_debug --mp_T 20 \
    --dalle_path vox_bert_text+draw_bs20_200k.pt
