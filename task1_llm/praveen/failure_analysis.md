# Task 1 — Sequence Model Failure Analysis

## Failure Case 1 — Strong Repetition in Greedy Decoding

**Generated behavior:**

The greedy sample produced nearly the same TinyStories-style passage twice, beginning with:

> "Once upon a time, there was a little girl named Lily..."

and repeating the same sequence involving the park and a box of colorful pictures.

**Failure type:** Repetition / decoding collapse

**Observation:**  
Greedy decoding always selects the highest-probability next character. This caused the model to repeatedly fall into a high-probability story template rather than generating diverse continuations.

---

## Failure Case 2 — Semantic Inconsistency

**Generated snippet:**

> "You have to be careful and eat some fresh and snacks on the wall."

**Failure type:** Semantic incoherence

**Observation:**  
The sentence is locally grammatical in parts, but the complete meaning is inconsistent. The phrase "snacks on the wall" does not fit the surrounding story context. The model learned common word and sentence patterns without reliably preserving higher-level meaning.

---

## Failure Case 3 — Object / World-Logic Error

**Generated snippet:**

> "It is a car full of trees. It is a toy can car."

**Failure type:** Hallucination / loss of coherence

**Observation:**  
The generated text combines familiar nouns and sentence templates in ways that do not form a coherent real-world description. The shorter 128-character context and smaller model capacity may make longer-range semantic consistency more difficult.

---

## Overall Observation

The model learned TinyStories-style grammar and short narrative patterns well, but generation still shows two characteristic weaknesses.

Greedy decoding is highly deterministic and can repeat common templates. Temperature sampling increases diversity, but it also produces more semantic inconsistencies.

The relatively high character repeated 4-gram rate is consistent with the repetition visible in generated text.
