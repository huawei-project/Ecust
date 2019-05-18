# train_exp_*.txt 0 samples.
- train_exp_{i}.txt denotes i-th random experiment.

# pairs_exp_*.txt 11592 samples, id: [1-63], band: [550,570,...,990], multispectral, non-obtructive.
- Each line of pairs_exp_*.txt includes two input image paths and their corresponding id.
- pairs_exp_{i}.txt denotes i-th random experiment.
- Positive samples are acquired by randomly choosing n_pos=4 pairs of images which have the same band and id. (n_sample = n_id * n_band * n_pos)
- Negative samples are acquired by randomly choosing n_neg=4 pairs of images which have the same band and different id. (n_sample = n_id * n_band * n_neg)

