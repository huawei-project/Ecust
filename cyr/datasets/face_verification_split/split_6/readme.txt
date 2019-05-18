# train_exp_*.txt 0 samples.
- train_exp_{i}.txt denotes i-th random experiment.

# pairs_exp_*.txt 756 samples, id: [1-63], rgb, non-obtructive.
- Each line of pairs_exp_*.txt includes two input image paths and their corresponding id.
- pairs_exp_{i}.txt denotes i-th random experiment.
- Positive samples are acquired by randomly choosing n_pos=6 pairs of images which have the same band and id. (n_sample = n_id * n_pos)
- Negative samples are acquired by randomly choosing n_neg=6 pairs of images which have the same band and different id. (n_sample = n_id * n_neg)

