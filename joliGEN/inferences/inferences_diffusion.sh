cd ~/vuong/augmentare/joliGEN/scripts
python3 gen_single_image_diffusion.py \
     --model-in-file ~/vuong/augmentare/joliGEN/checkpoints/new_diffusion_male_gray_2_female_gray/latest_net_G_A.pth \
     --img-in ~/vuong/augmentare/joliGEN/dataset/new_diffusion_male_gray_2_female_gray/trainA/img/000221.jpg \
     --mask-in ~/vuong/augmentare/joliGEN/dataset/new_diffusion_male_gray_2_female_gray/trainA/bbox/000221.jpg \
     --dir-out ~/vuong/augmentare/joliGEN/inferences/diffusion_male_gray_2_female_gray/new \
     --nb_samples 4 \
     --img-width 256 \
     --img-height 256