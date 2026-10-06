# DATA266 Lab 1 - Pair 25

This repository contains the team submission for DATA266 Lab 1. Naveen and Praveen maintain independent implementations under separate folders for all three tasks.

## Naveen (SID 0171)

- Task 1: `task1_llm/naveen/`
- Task 2: `task2_sentiment/naveen/`
- Task 3: `task3_gan/naveen/`

## Praveen (SID 8511)

- Task 1: `task1_llm/praveen/`
- Task 2: `task2_sentiment/praveen/`
- Task 3: `task3_gan/praveen/`

Each member folder contains the relevant implementation, metrics, outputs, analysis, and results summary.

## Smoke test

From the repository root:

```bash
python task3_gan/praveen/src/train.py --config task3_gan/praveen/src/config.json --output-root task3_gan/praveen/smoke_outputs --smoke-test --synthetic-smoke-data --device cpu
