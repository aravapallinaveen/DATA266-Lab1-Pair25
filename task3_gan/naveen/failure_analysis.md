# Task 3 Failure / Limitation Analysis

## Summary

The final epoch-150 CycleGAN preserved source content better than it preserved visual cleanliness. In the 30-sample blinded audit, two independent raters gave a mean content-preservation score of 4.033/5, compared with 3.667/5 for style quality and 3.433/5 for artifact quality. The overall human-audit score was 3.711/5.

Inter-rater agreement was reasonably consistent overall. The mean quadratic-weighted Cohen's kappa across style, content, and artifact ratings was 0.674, with mean exact agreement of 57.78%.

## Observed limitations

Artifact quality was the weakest human-rated dimension. Some translations preserved the source scene while still introducing visible texture or rendering artifacts. The main limitation was therefore not complete content loss, but producing a cleaner target-domain appearance.

The two translation directions were also asymmetric. Monet-to-Photo (A2B) received higher human ratings than Photo-to-Monet (B2A) for style, content preservation, and artifact quality.

A2B had higher generative precision but lower recall, while B2A had lower precision and higher recall. This suggests A2B outputs were concentrated in a narrower region of the target distribution, while B2A covered more variation with lower precision.

## Stability and reconstruction

The selected checkpoint completed evaluation with zero recorded NaN and Inf values. Cycle-reconstruction L1 remained low in both directions, which is consistent with the stronger human content-preservation score. Low reconstruction error, however, did not guarantee artifact-free translations.

## Possible improvements

Future experiments would focus on reducing visible artifacts while maintaining content consistency. Possible improvements include alternative upsampling methods, discriminator regularization, replay-buffer tuning, and checkpoint selection using both quantitative metrics and a fixed qualitative sample set.

Human-audit evidence is stored in:

- `outputs/human_audit/human_audit_30_samples_completed.csv`
- `outputs/human_audit/human_audit_summary.csv`
- `outputs/human_audit/human_audit_by_direction.csv`
