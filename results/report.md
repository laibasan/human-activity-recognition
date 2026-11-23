# Human Activity Recognition — Training Report

## Model
- Architecture: BiLSTM + Attention → Linear(128, 6)
- Trainable parameters: 46,982
- Device: cpu

## Training
- Initial loss: 1.8049  (log 6 ≈ 1.7918)
- Best epoch: 5
- Best validation loss: 0.1208
- Train loss at best: 0.1241
- Validation accuracy at best: 96.44%
- Training time: 35.5 minutes

## Test Results
- Test accuracy: 91.45%
- Test samples: 2947

### Per-Class Accuracy
  - WALKING                87.70%
  - WALKING_UPSTAIRS       98.30%
  - WALKING_DOWNSTAIRS     97.38%
  - SITTING                74.75%
  - STANDING               90.98%
  - LAYING                100.00%

### Attention Peakedness (mean of max alpha)
  - WALKING               0.2978
  - WALKING_UPSTAIRS      0.2364
  - WALKING_DOWNSTAIRS    0.2784
  - SITTING               0.0144
  - STANDING              0.0184
  - LAYING                0.0106
