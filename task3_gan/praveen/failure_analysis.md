# Task 3 Failure Analysis

## Scope

This analysis describes limitations observed in the final CycleGAN evaluation.
No additional training was performed.

## Observed metric limitations

The final local evaluation produced:

- Overall FID: 112.774391
- Overall MiFID: 0.423197
- A2B content cosine: 0.8229490374
- B2A content cosine: 0.7893867940

The FID values indicate that the generated distributions were not yet close
to the corresponding real-image distributions. The lower B2A content-cosine
score suggests that Photo-to-Monet translation preserved source content less
consistently than Monet-to-Photo translation.

## Human-audit findings

Thirty deterministic samples were reviewed by two independent human raters:
15 A2B and 15 B2A samples.

The artifact scale is severity-based:

- 1 = no visible artifacts
- 5 = severe visible artifacts
- Lower artifact scores are better

Aggregate human-audit results are stored in:

- `outputs/human_audit_30_samples.csv`
- `outputs/human_audit_summary.csv`

The overall mean artifact score was 3.933333. This indicates that visible
artifacts remained an important limitation in the reviewed outputs.

## Reproducibility limitation

The final 300-epoch checkpoint is preserved as
`outputs_final_multiscale/checkpoints/epoch_0300.pt`.

The original console log for the final 300-epoch run was not present in the
recovered workspace. The available 200-epoch baseline and 250-epoch tuned logs
are copied unchanged under `reproducibility/raw_logs/` with accurate filenames.
