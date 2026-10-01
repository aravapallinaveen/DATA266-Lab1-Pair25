# Task 2 Experimental I — Raw Log Recovery Note

The official Experimental I training run completed successfully for all 8 epochs.

Preserved evidence:
- Best checkpoint: experimental_stacked_bigru_best.pt
- Best epoch: 7
- Best validation loss: 0.161809
- Validation accuracy at best checkpoint: 0.943161
- Validation macro-F1 at best checkpoint: 0.943160
- Last checkpoint: experimental_stacked_bigru_last.pt
- Complete 8-row training history CSV:
  experimental_stacked_bigru_training_history.csv

After the official run completed, the training cell was accidentally launched again.
The training function opened the same raw-log filename in write mode, overwriting the
original raw log. The accidental rerun was manually interrupted before completing
epoch 1.

The surviving checkpoints and training-history CSV are from the completed official run.
The interrupted raw log is retained without further editing.
