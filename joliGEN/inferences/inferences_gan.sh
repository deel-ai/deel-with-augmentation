cd ~/vuong/augmentare/joliGEN/scripts
python3 gen_single_image.py \
     --model-in-file ~/vuong/augmentare/joliGEN/checkpoints/male_gray2blond_cut/latest_net_G_A.pth \
     --img-in ~/vuong/augmentare/joliGEN/dataset/male_gray2blond/trainA/000221.jpg \
     --img-out ~/vuong/augmentare/joliGEN/inferences/male_gray2blond_cut/fake_000221.jpg \