python3 test.py --name test_vox_text_roberta \
    --image_text_folder data/mmvoxceleb \
    --dataset video_text --text_seq_len 50 \
    --which_tokenizer simple --use_html --num_visuals 0 \
    --num_targets 8 --frame_num 8 --frame_step 4 \
    --image_size 128 \
    --iters 1 --batch_size 16 --n_per_sample 4 --n_sample 1 \
    --no_debug --mp_T 20 --dalle_path vox_bert_text_txtdrop_roberta_bs24_112k.pt \
    --fixed_language_model roberta-large \
    --description "A girl."

# NOTE: --description="A person has no hair." or "A person wears spectacles." or "A person is youthful."