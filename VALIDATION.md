# Validation record

Checked on 15 October 2026 in the prepared Windows environment with an NVIDIA
GeForce RTX 5050 Laptop GPU, PyTorch 2.11.0+cu128, and torchvision 0.26.0+cu128.

| Check | Result |
|---|---|
| Notebook format and Python syntax | All 13 notebooks passed |
| Fresh-kernel GPU execution with tiny smoke datasets | All 13 notebooks passed across the recorded runs |
| Normal `QUICK=True` GPU demonstrations | All 11 training notebooks, 02–12, passed |
| CPU fallback | Notebook 08 passed, including disabled AMP, accumulation, save, and resume |
| Deployment | Notebook 12 passed weight reload parity, PNG preprocessing parity, a separate prediction process, and `torch.export` save/load parity |
| Visual inspection | Beginner learning curves, CNN feature maps, Grad-CAM galleries, and calibration plots reviewed |
| Source notebooks | Outputs and execution counts remain cleared for teaching |

Detailed records are in `validation/*.json`. The consolidated
[validation/summary.json](validation/summary.json) lists passing checks by notebook.
Executed notebooks with real plots are in `validation/executed/`. These are examples
of actual runs; their values are not expected answers or guaranteed accuracy targets.

Normal demonstration results included a 76.9% linear-classifier test accuracy and
an 81.2% CNN test accuracy on each lesson's selected 1,000-image Fashion-MNIST test
subset. The transfer-learning demonstration trained on 800 CIFAR-10 images and
achieved 78.3% on its selected 300-image test subset. These limited experiments
show that the demos learn; they are not full-dataset benchmark claims.

The default data pipelines, plots, controlled comparisons, temperature fitting,
Grad-CAM, checkpoint recovery, and standalone inference were executed. The
`QUICK=False` longer experiments, the optional pretrained ViT/photo branch,
arbitrary user-provided datasets, and every exercise variation were not executed.

## Resolved setup issues

- The initial official CIFAR-10 download was too slow and its disposable validation
  kernel was stopped. The archive was obtained from a public mirror documented by
  [Volcengine](https://docs.volcengine.com/docs/ml-platform/quick-start-for-algorithm-engineers?lang=en),
  and its MD5 was checked against torchvision's expected
  `c58f30108f718f92721af3b95e74349a` before use. Both the smoke and normal transfer-learning
  runs subsequently passed. Notebook code still uses torchvision's standard dataset loader.
- The CPU test harness initially used an empty CUDA device list, which behaved
  inconsistently in this Windows setup. It now uses `CUDA_VISIBLE_DEVICES=-1`;
  the CPU training and resume check subsequently passed.
- The installed optional Triton package prints a missing CUDA Toolkit warning,
  but ordinary PyTorch CUDA training and export completed successfully. The course
  does not require Triton compilation.

Historic failed executed copies are retained locally for diagnosis; they are
superseded by the passing records and are excluded from the downloadable course ZIP.
Some validation runs shared the GPU, so their timings are not controlled speed benchmarks.
