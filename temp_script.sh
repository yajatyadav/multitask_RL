for i in {1..5}; 
do CUDA_VISIBLE_DEVICES=0 uv run evaluation/eval_libero_BERT_classifier_best_of_N.py --env_name 'libero_spatial-pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate|libero_spatial-pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate' --scene libero_spatial --num_demos 25 --n_vals 1 4 16 64 128 256 512
done;