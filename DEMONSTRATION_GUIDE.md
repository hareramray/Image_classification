# Teaching by doing

Use one notebook per session for a full course. For each section: state the question,
ask learners to predict the result, run the cell, inspect the output, then change one setting.
This turns a sequence of executions into an experiment that people can explain.

## Before a live session

1. Start Jupyter from the project root and select the course kernel.
2. Run notebook 00 to verify the actual compute device.
3. Run the lesson once with the intended settings to cache its data and weights.
4. Use `QUICK=True`; train a longer model beforehand only if you want stronger examples.
5. Keep an executed copy in `validation/executed/` available for reference. Label smoke outputs as execution checks.
6. Restart the teaching notebook so variables begin in a known state.
7. Reserve time for reading failures and asking what the model has learned.

Do not promise a particular accuracy or that every modification improves it.
Changing one factor at a time and interpreting an unsuccessful experiment are valuable demonstrations.

## Suggested schedules

| Format | Notebook sequence | Emphasis |
|---|---|---|
| 90-minute introduction | Selected cells from 00, 01, 02, and 04 | Pixels → batches → training → predictions |
| Half-day workshop | 00–04, selected evaluation cells from 05 | Beginner foundation and a working CNN |
| Two-day practical workshop | Day 1: 00–05; Day 2: 06–09 and 12 | Full classification workflow and custom images |
| Full course | 00–12 over 10–13 sessions | All exercises, experiments, calibration, ViT, capstone |

The shorter formats require selecting cells, preparing downloads, and running longer training
beforehand. They are not promises that every full notebook fits into the allotted time.

## Demonstration cues

| Lesson | Ask before running | Make this visible | One change to try |
|---|---|---|---|
| 00 | What does `[8, 3, 32, 32]` mean? | Device, shapes, gradients | Change a logit or learning rate |
| 01 | Could the same person's photos appear in both splits? | Images, class counts, disjoint indices | Change batch size |
| 02 | What score can we achieve without learning? | Baseline versus trained loss | Change learning rate |
| 03 | Why add a hidden layer? | Forward shapes and learning curves | Change hidden width |
| 04 | What does a filter respond to? | Filters and feature maps | Change channel count |
| 05 | Which classes are confused? | Confusion matrix and actual mistakes | Change augmentation strength |
| 06 | Which weights are changing? | Trainable count and validation comparison | Change fine-tuning learning rate |
| 07 | Can a duplicate leak into validation? | Folder layout, audit, split manifest | Add a new class with enough images |
| 08 | Does a larger batch always need more GPU memory? | Accumulation, scaler, checkpoint state | Change microbatch/accumulation settings |
| 09 | Does a heatmap prove a prediction is correct? | Saliency, Grad-CAM, occlusion | Explain another class logit |
| 10 | How does an image become a token sequence? | Patch grid, shape trace, learning curves | Change patch size to a divisor of 28 |
| 11 | Are 90% confidence predictions correct 90% of the time? | Reliability plot and corruption results | Declare new noise severities before evaluation |
| 12 | What else must travel with the weights? | Reload parity and command-line JSON | Measure complete input-to-output latency |

## A useful rhythm for each lesson

Spend a few minutes on the learning objective, inspect inputs before training, and run the first
batch slowly enough to explain its shape. During training, connect changes in loss to the update
rule. After training, show individual successes and failures alongside aggregate metrics.
Finish with one exercise that the learner edits and runs themselves.

Validation is for choosing the model; test is for the final report. If a test result inspires a new
experiment, treat that result as exploratory and reserve fresh held-out data for final evaluation.
Keep related images together when splitting your own data.

## Capstone presentation rubric

| Area | Evidence | Suggested points |
|---|---|---|
| Problem and data | Clearly defined classes, representative images, provenance | 15 |
| Splitting and audits | Duplicate/group leakage checks, class coverage, untouched test | 20 |
| Baseline and experiments | Reproducible baseline, controlled validation comparison | 20 |
| Evaluation | Per-class metrics, mistakes, confidence and corruption limitations | 20 |
| Reusable inference | Saved preprocessing/class mapping, reload parity, image prediction | 15 |
| Explanation | Clear live demonstration and concrete limitations | 10 |

Reward a reproducible, well-explained experiment over a high accuracy number without a trustworthy split.
