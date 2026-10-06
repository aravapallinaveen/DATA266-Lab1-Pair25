# Praveen 8511 versus Naveen - Task 3 distinction record

Praveen's model is independently configured and does not reproduce Naveen's architecture or hyperparameters.

| Area | Naveen | Praveen 8511 |
|---|---|---|
| Generator channels | 64-128-256 | 48-96-192 |
| Residual blocks | 9 | 6 |
| Upsampling | Transposed convolution | Nearest-neighbor plus convolution |
| Discriminator channels | 64-128-256-512 | 48-96-192-384 |
| Discriminator regularization | Not specified | Spectral normalization |
| Adversarial objective | LSGAN/MSE | BCE with logits |
| Cycle weight | 10 | 12 |
| Identity weight | 5 | 2 |
| Optimizer | Adam | AdamW |
| Learning rate | 2e-4 | 1e-4 |
| Betas | 0.5, 0.999 | 0.5, 0.99 |
| Batch size | 1 | 2 |
| Seed | 171 | 8511 |
| Replay buffer | Not specified | 50 images |
| Learning-rate schedule | Constant then linear decay | Constant then cosine decay |

The shared task requirements remain satisfied: two generators, two discriminators, unpaired domains, adversarial and cycle-consistency losses, image translation, stability analysis, required metrics, direct Kaggle inference, and human audit.


## Final selected configuration

The submitted final model uses 64 base channels, 9 generator residual blocks,
a MultiScaleDiscriminator, no spectral normalization, LSGAN/MSE adversarial
loss, Adam with learning rate 2e-4, cycle weight 10, identity weight 5,
batch size 2, replay buffer 50, and 300 epochs.
