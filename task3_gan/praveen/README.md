
## Final multiscale configuration

The final Task 3 model used:

- 64 base channels
- 9 generator residual blocks
- MultiScaleDiscriminator
- Spectral normalization disabled
- LSGAN objective with MSE loss
- Adam optimizer with learning rate 0.0002
- Cycle-consistency weight: 10
- Identity-loss weight: 5
- Batch size: 2
- Replay buffer size: 50
- 300 training epochs
- Seed: 8511

The final checkpoint is preserved as `outputs_final_multiscale/checkpoints/epoch_0300.pt`.
