# Task 2 Failure Analysis — Naveen Aravapalli

## Scope

This file summarizes the manually reviewed 20 errors selected from the saved test predictions for the Experimental I model (`experimental_stacked_bigru`). The worksheet contains exactly 5 confident false positives, 5 confident false negatives, 5 near-threshold errors, and 5 slice-specific failures.

The detailed observations and proposed fixes below are taken from the completed manual review worksheet (`outputs/manual_20_error_review_completed.csv`).

## Selection Summary

| Category | Cases |
|---|---:|
| Confident false positives | 5 |
| Confident false negatives | 5 |
| Near-threshold errors | 5 |
| Slice-specific failures | 5 |
| **Total** | **20** |

## Confident False Positives

### Test index 29330
- **Ground truth / prediction:** 0 → 1
- **P(positive):** 0.999938
- **Processed length:** 17
- **Review excerpt:** Wow love the place and everything is very clean and new!\n\nGreat place to come and relax worth a try!\n\nCheers,\n\nEric Van Nguyen\nVisited April 2012
- **Error type:** noisy label
- **Observation:** The review text is overwhelmingly positive, but the ground truth label is negative, confusing the model.
- **Testable fix:** Implement cross-validation confident learning to identify and prune mislabeled training instances.

### Test index 408
- **Ground truth / prediction:** 0 → 1
- **P(positive):** 0.999878
- **Processed length:** 53
- **Review excerpt:** Though I'm a Copper enthusiast when it comes to getting my Indian fix in Charlotte, I'd heard that Maharani was a cheaper but tasty option, so we ordered from there a few nights ago. \n\nCo…
- **Error type:** mixed sentiment
- **Observation:** The review is lukewarm but contains strong positive keywords ("amazing", "great"), causing a false positive.
- **Testable fix:** Train with aspect-based sentiment features to better weigh mild negatives against positive keywords.

### Test index 501
- **Ground truth / prediction:** 0 → 1
- **P(positive):** 0.999777
- **Processed length:** 41
- **Review excerpt:** Laser quest is a fun experience for kids of all ages, but the adults that play at this establishment, honestly kind of creep me out. The best time to bring your children here is during the …
- **Error type:** conflicting context
- **Observation:** Positive framing of the game itself is overridden by a specific severe negative ("adults creep me out"), which the model missed.
- **Testable fix:** Use self-attention to capture the semantic weight of strong negative social cues over general descriptions.

### Test index 8426
- **Ground truth / prediction:** 0 → 1
- **P(positive):** 0.999665
- **Processed length:** 213
- **Review excerpt:** Saturday / Sunday AYCE brunch\n\nIn true las vegas fashion, you get a flat rate to eat your heart out. \n\nFor Strip food, main menu items seem reasonably priced and the brunch is cheap at …
- **Error type:** aspect sentiment imbalance
- **Observation:** The reviewer praises the service but criticizes the core product (food); the model over-indexed on positive service keywords.
- **Testable fix:** Apply aspect-based sentiment analysis (ABSA) to decouple food sentiment from service sentiment.

### Test index 18794
- **Ground truth / prediction:** 0 → 1
- **P(positive):** 0.999598
- **Processed length:** 26
- **Review excerpt:** You can do yourself a huge favor and have all required documents in order before you arrive. You will find that this will speed up your time here. Also, call them and ask questions before y…
- **Error type:** neutral phrasing interpreted as positive
- **Observation:** The review offers neutral practical advice ("do yourself a huge favor") without praising the business.
- **Testable fix:** Expand the training data with neutral, advice-based reviews to separate subjective praise from objective tips.

## Confident False Negatives

### Test index 30793
- **Ground truth / prediction:** 1 → 0
- **P(positive):** 0.000038
- **Processed length:** 37
- **Review excerpt:** This place is so much better since they changed owners.\n\nMy wife and I went when it was the old owners, it was terrible. We waited forever and the food never came before we walked out. Pe…
- **Error type:** temporal sentiment shift
- **Observation:** The review heavily describes a past negative experience before clarifying the current experience is positive.
- **Testable fix:** Test sentence-level recency-aware pooling that gives greater weight to explicit update/current-experience statements.

### Test index 22807
- **Ground truth / prediction:** 1 → 0
- **P(positive):** 0.000040
- **Processed length:** 47
- **Review excerpt:** EDIT: They really did change the service up since I last posted this.\n\nHorrible service.\n\nUsed to be my favorite pizza in the city (at a reasonable price), but I'm rethinking that. We j…
- **Error type:** update/revision ambiguity
- **Observation:** The text explicitly describes an altercation and calls the service "horrible," yet the ground-truth label is positive.
- **Testable fix:** Detect EDIT/UPDATE markers and separately encode revised and original portions before classification.

### Test index 20061
- **Ground truth / prediction:** 1 → 0
- **P(positive):** 0.000052
- **Processed length:** 256
- **Review excerpt:** Ever wonder what to do if you have lots of extra garbage or recyclables and either can't fit them all in your bins or missed bulk trash pickup day? Alternatively, are you looking for someth…
- **Error type:** domain-specific vocabulary
- **Observation:** The text contains inherently negative words ("garbage", "stinks") that are literal descriptions of a waste facility.
- **Testable fix:** Test context-aware sentence pooling learned from scratch so words such as "garbage" and "stinks" are interpreted within the waste-facility context.

### Test index 9871
- **Ground truth / prediction:** 1 → 0
- **P(positive):** 0.000122
- **Processed length:** 129
- **Review excerpt:** After a major screw up by me, booking the room for the 22nd and driving up to Vegas on the 21st, it was not until I hit state line when The Jennifer mentioned today is the \""21st\"" until …
- **Error type:** self-deprecation
- **Observation:** The review starts with a long negative story about the user's own mistake before praising the hotel.
- **Testable fix:** Test target-aware or sentence-level attention learned from scratch to distinguish complaints about the reviewer’s own mistake from sentiment toward the hotel.

### Test index 26172
- **Ground truth / prediction:** 1 → 0
- **P(positive):** 0.000136
- **Processed length:** 77
- **Review excerpt:** I had to eat here at least once, and would eat here more often if it were in a more convenient location for me. Have you ever wanted to try real ramen? News flash -- the 99 cent packages of…
- **Error type:** contrastive sarcasm
- **Observation:** The user praises the restaurant by aggressively attacking instant ramen; the model failed to grasp the rhetoric.
- **Testable fix:** Train on a dataset rich in sarcastic or contrastive structures to capture complex sentence-level pragmatics.

## Near-Threshold Errors

### Test index 20109
- **Ground truth / prediction:** 0 → 1
- **P(positive):** 0.500004
- **Processed length:** 22
- **Review excerpt:** Food was good but, not great. Location is very cool, in the town center. (think Big White columned building with southern looking square)\n\nNo lunch specials-you paid dinner prices.
- **Error type:** balanced mixed sentiment
- **Observation:** The review contains an almost equal mix of mild positives and negatives, pushing the probability right to 0.5.
- **Testable fix:** Test sentence-level attention or calibrated decision thresholds on mixed-sentiment reviews while retaining binary classification.

### Test index 36081
- **Ground truth / prediction:** 0 → 1
- **P(positive):** 0.500132
- **Processed length:** 58
- **Review excerpt:** Summary: great location, great micro brew, really good appetizers. So why the 2 stars? Horrible service. \n\nAdvice: if you are waiting for a show, this is not the place to go if you want t…
- **Error type:** explicit rating override
- **Observation:** The user explicitly states "why the 2 stars" and balances praise with severe criticism of service.
- **Testable fix:** Add a feature extraction step that specifically looks for explicit text-based ratings (e.g., "2 stars") to weight the prediction.

### Test index 3993
- **Ground truth / prediction:** 1 → 0
- **P(positive):** 0.499149
- **Processed length:** 10
- **Review excerpt:** The place is open 24/7, is pretty quick with orders, is priced very competitively. I have yet to find a better place for tortas!
- **Error type:** complex negation
- **Observation:** The phrase "yet to find a better place" is a strong positive, but the model likely parsed "yet to find" negatively.
- **Testable fix:** Augment the training data with examples of complex negations and idioms to improve syntactic comprehension.

### Test index 861
- **Ground truth / prediction:** 1 → 0
- **P(positive):** 0.499005
- **Processed length:** 218
- **Review excerpt:** Summer time is supposed to be \""healthy time\"", so I am trying to watch what I eat. Now, I will always be a big guy (not \""two fat twins on their matching motorcycles\"" big, but you kno…
- **Error type:** verbose narrative dilution
- **Observation:** The core positive sentiment is diluted by long tangents about diets and minor complaints, dragging the probability down.
- **Testable fix:** Implement an extractive summarization attention layer to focus the classifier on concluding sentences.

### Test index 13662
- **Ground truth / prediction:** 0 → 1
- **P(positive):** 0.501028
- **Processed length:** 70
- **Review excerpt:** The Bistro Buffet at the Palms is not among my top ten favorite buffets in Las Vegas, as the food can be tepid to cold, the meats selection can be dry, and the specialty coffee is served fr…
- **Error type:** conditional recommendation
- **Observation:** The review recommends one specific food section but concludes with a general rejection ("give it a pass").
- **Testable fix:** Add positional weighting to the final sentences of long reviews to capture overarching conclusions.

## Slice-Specific Failures

### Test index 9477
- **Ground truth / prediction:** 1 → 0
- **P(positive):** 0.000180
- **Processed length:** 256
- **Review excerpt:** Before getting started, I'd like to point out that I am currently sitting in the Palms Casino, while watching my degenerate, gambleholic, asshole-of-a-friend Pete grab his ankles, as the cr…
- **Error type:** excessive profanity
- **Observation:** The review is saturated with profanity and gambling losses, which the model strongly correlates with negative sentiment.
- **Testable fix:** Use domain adaptation for Vegas/Nightlife reviews where profanity is often used in a positive, entertaining context.

### Test index 30958
- **Ground truth / prediction:** 1 → 0
- **P(positive):** 0.000212
- **Processed length:** 181
- **Review excerpt:** Look, we all know Cox sucks. In fact they are a terrible business and their practices are laughable. However they are a necessary evil if you want Internet that isn't garbage that can handl…
- **Error type:** brand vs. local sentiment
- **Observation:** The user spends 80% of the review ranting against the corporate brand but highly praises the local manager.
- **Testable fix:** Use entity-level sentiment analysis to separate sentiment directed at corporate vs. the specific location.

### Test index 29494
- **Ground truth / prediction:** 1 → 0
- **P(positive):** 0.000411
- **Processed length:** 141
- **Review excerpt:** UPDATED. \n\nMy initial very frustrated and dramatic review read as follows:\n\nBililng practices are at best negligent and at worst fraudulent. \n\nI started going to the studio per a grou…
- **Error type:** updated review format
- **Observation:** The text contains an old angry review appended with a new positive update. The model counted the old negative words.
- **Testable fix:** Add a preprocessing step that detects "UPDATED" markers and gives significantly higher weight to the new text.

### Test index 5752
- **Ground truth / prediction:** 0 → 1
- **P(positive):** 0.999198
- **Processed length:** 207
- **Review excerpt:** NOTE: This was a 4-star review, but the food quality and ESPECIALLY customer service have gone down the tubes. See update below.\nComplaining I can't find a good meatball sub in Phoenix, I …
- **Error type:** updated review format (reversed)
- **Observation:** The reviewer appended a brief negative update at the top but left the massive glowing original review intact below it.
- **Testable fix:** Implement a decay function that prioritizes the first few lines of text if explicit rating revisions are detected.

### Test index 21813
- **Ground truth / prediction:** 1 → 0
- **P(positive):** 0.001149
- **Processed length:** 227
- **Review excerpt:** Our flight arrived to Vegas earlier than excepted, so we expected our room not to be ready. When we arrived at the hotel on May 19th, the front desk girl offered us a room that was ready on…
- **Error type:** high tolerance reviewer
- **Observation:** The reviewer objectively lists multiple severe service failures yet subjectively forgives them and leaves a positive rating.
- **Testable fix:** Test sentence-level attention that emphasizes the reviewer's final explicit recommendation ("would stay there again") over earlier incident descriptions.


## Recurrent Patterns in the Reviewed Errors

Across the completed worksheet, the recurring error types include mixed or aspect-conflicting sentiment, temporal/update reversals, target/entity confusion, domain-specific wording, complex negation, long narrative dilution, explicit rating/recommendation overrides, and reviewer-language patterns such as profanity or high tolerance for service failures.

The proposed fixes in the worksheet are intentionally testable: examples include sentence-level or target-aware attention/pooling, explicit handling of EDIT/UPDATE markers, stronger context modeling learned from scratch, targeted augmentation for negation/idiomatic structures, and features for explicit ratings or recommendations.

## Artifact

The complete review with full review text is stored in:

`outputs/manual_20_error_review_completed.csv`
