# ViT-B/16 from scratch: CrisisMMD image classifier (informative vs. not informative)

The image-only, no-pre-training baseline for RQ3. It uses agreed-label pairs only and the event split.

| Split | Events |
|---|---|
| train | hurricane_harvey, hurricane_irma, hurricane_maria, srilanka_floods |
| val (model selection) | mexico_earthquake, iraq_iran_earthquake |
| test (held-out event) | california_wildfires |

## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. Get the agreed-label pairs

The Hugging Face copy (`QCRI/CrisisMMD`) has only one label per row, so it can't tell you whether the text and image labels agree. Download **`crisismmd_datasplit_agreed_label.zip`** from the CrisisMMD page (crisisnlp.qcri.org/crisismmd) and unzip it:

```bash
unzip crisismmd_datasplit_agreed_label.zip -d ./agreed
```

The script pools the released train, dev and test files for the informative task (every pair in them has matching labels) and re-splits them by event. If a file has both `label_text` and `label_image` columns, the script also checks that they match. Images come from Hugging Face. On the first run they're resized (shorter side 256) and cached, and the script prints how often the HF label matches the agreed label as a sanity check.

## 3. Run

```bash
# smoke test (2 epochs, about 500 images), to check the pipeline end to end
python train.py --agreed_dir ./agreed --cache_dir ./cache_256 --out_dir ./runs/smoke --smoke

# full run; use 3 seeds and report mean ± std
for s in 0 1 2; do
  python train.py --agreed_dir ./agreed --cache_dir ./cache_256 \
                  --out_dir ./runs/vit_b16_scratch_s$s --seed $s
done
```


## Recipe (defaults)

| Setting | Value |
|---|---|
| Model | `vit_base_patch16_224`, `pretrained=False`, 2-class head (backbone = ViT-B/16, 86.6M params with the ImageNet head) |
| Optimizer | AdamW, peak LR 5e-4 × batch/512, wd 0.05 (none on norms, biases, pos-embed, cls), betas (0.9, 0.999) |
| Schedule | 200 epochs, 10 warmup epochs, cosine decay to 1e-6, grad clip 1.0 |
| Augmentation | RandomResizedCrop(224, scale 0.25–1), h-flip, RandAugment m9, random erasing 0.25 |
| Regularization | Mixup 0.8 / CutMix 1.0, label smoothing 0.1, stochastic depth 0.1 |
| Precision | bf16 autocast (fp16 + GradScaler if bf16 is unsupported) |
| Selection | Best validation macro-F1 checkpoint, evaluated once on test |


## Outputs (`--out_dir`)

- `metrics.json`: val and test accuracy, macro-F1, per-class F1, confusion matrix, best epoch
- `per_event_metrics.csv`: the same metrics for each event (val: the two earthquakes; test: California wildfires)
- `history.csv`: per-epoch train loss, val loss, val acc, val macro-F1
- `data_manifest.csv`: every pair used, with its split. Use it to check that teammates trained on the same subset
- `config.json`, `best.pt`: settings used and best weights
