# Task 1 — Sequence Model Failure Analysis

## Failure Case 1 — Semantic Inconsistency

**Generated snippet:**

> "She saw a big tree with a long tail on it. She wanted to pick it up and see what was inside."

**Failure type:** Loss of coherence / semantic inconsistency

**Observation:**  
The sentence is grammatically structured, but the described object behavior is not logically consistent. A tree is described as having a long tail and as something that can be picked up. This shows that the model learned common sentence patterns more strongly than physical or semantic consistency.

---

## Failure Case 2 — Nonsensical Phrase

**Generated snippet:**

> "He knew that he would always pay for a day of step."

**Failure type:** Broken semantics / incoherent phrase

**Observation:**  
The model produces a sentence with valid local grammar, but the phrase "pay for a day of step" has no clear meaning. This suggests that character-level prediction can create fluent-looking syntax without preserving higher-level meaning.

---

## Failure Case 3 — Template-Like Repetition

**Generated behavior:**

> "She wanted to ... and see what was inside."

Similar sentence structures appeared repeatedly in greedy generation.

**Failure type:** Repetition / low-diversity decoding

**Observation:**  
Greedy decoding repeatedly selects the highest-probability next character and tends to fall into common TinyStories-style sentence templates. Temperature sampling produced more variation, while greedy decoding was more deterministic and repetitive.

---

## Overall Observation

The model produces mostly grammatical TinyStories-style text, but occasional semantic inconsistencies and repetitive sentence templates remain. Temperature sampling improves diversity compared with greedy decoding, although it can also introduce less coherent continuations.
