# Training data layout

This directory is git-ignored. Put your own labelled videos here to train or
evaluate the pose-based detector.

```
training_data/
├── seizure/
│   ├── videos/       # confirmed seizure recordings
│   └── processed/    # extracted pose sequences (generated)
└── normal/
    ├── videos/       # baseline recordings: sleeping, waking, daytime activity
    └── processed/    # extracted pose sequences (generated)
```

Extract pose sequences from both folders:

```bash
python scripts/extract_pose_sequences.py \
    --seizure-dir training_data/seizure/videos \
    --normal-dir  training_data/normal/videos \
    --output-dir  training_data
```

Then follow [docs/CUSTOM_MODEL_TRAINING.md](../docs/CUSTOM_MODEL_TRAINING.md).

Videos of a patient are sensitive health data. Keep them out of version
control and out of any shared storage you do not fully control.
