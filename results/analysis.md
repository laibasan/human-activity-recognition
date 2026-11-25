# Task 6: Analysis and Reflection

This document contains the analytical findings and answers to Task 6 for the Human Activity Recognition (HAR) project using the UCI HAR dataset.

---

## 6.1 Validation Split by Person vs. Random Windows

**Question:** Why is the validation set chosen by person rather than by picking random windows? *(Hint: neighbouring windows overlap by 50%.)*

**Answer:**  
1. **Preventing Data Leakage:** Consecutive windows in the UCI HAR dataset are extracted using a 50% temporal overlap (1.28 seconds of shared signal between adjacent 2.56-second windows). If windows were randomly partitioned, adjacent overlapping windows from the same subject performing the same activity would be split across training and validation sets. The network could simply memorize local temporal continuations rather than learning generalized activity patterns, producing artificially inflated, over-optimistic performance metrics.
2. **Evaluating Subject Generalization:** Movement dynamics (stride frequency, gait speed, pelvic oscillation, body mass) vary significantly across individuals. By isolating four specific volunteers (`VAL_SUBJECTS = [3, 15, 25, 30]`) entirely from training, we enforce a strict subject-independent evaluation. This mirrors clinical and consumer deployment, where wearables must generalize to new, unseen users without recalibration.

---

## 6.2 Confusion Between Activities

**Question:** Which two activities are confused most often? Using the plots from Task 1.3, explain why they are hard to separate from a waist-worn phone.

**Answer:**  
- **Most Confused Pair:** **SITTING** and **STANDING**.  
  - Out of 491 true Sitting test windows, **122 were misclassified as Standing**.
  - Out of 532 true Standing test windows, **48 were misclassified as Sitting**.
  - Across all other activities, cross-category errors were negligible (e.g., Laying had 0 errors).

- **Physical Reason:**  
  Looking at `example_windows.png` (Task 1.3), both Sitting and Standing are static, non-periodic postures characterized by nearly zero dynamic acceleration and negligible angular velocity across all three gyroscope axes ($\approx 0\text{ rad/s}$).  
  When a smartphone is mounted at the waist, the sensor's static reading is dominated solely by gravity ($\approx 1g$). For a person sitting upright with good posture versus standing upright, the tilt angle of the pelvis relative to gravity is nearly identical. Subtle variations in posture (slouching, leaning back, phone tilt in pocket) completely blur the boundary between the two states, making separation without posture transition cues notoriously difficult.

---

## 6.3 Perfect Recognition of LAYING

**Question:** LAYING is usually recognised almost perfectly. Which of the 9 signals makes it easy, and why?

**Answer:**  
- **Accuracy:** LAYING achieved **100.00% per-class accuracy** (537 out of 537 test windows correctly identified with 0 false positives).
- **Key Signal:** **`total_acc_x`** (Total acceleration along the X-axis).
- **Physical Reason:**  
  The sensor coordinate frame aligns with the smartphone: in an upright waist mount (standing, sitting, walking), the body's vertical axis points downward, so Earth's gravitational acceleration ($1g \approx 9.8\text{ m/s}^2$) projects heavily onto the phone's Y or Z axis, while the X-axis measures near zero static acceleration.  
  When an individual is laying down (supine, prone, or lateral), the body axis rotates by 90 degrees relative to gravity. Consequently, the $1g$ gravitational vector projects directly onto the orthogonal phone axis (`total_acc_x` registers close to $1g$). This macroscopic, static coordinate shift is unique to laying down and remains invariant across individuals, allowing the linear readout layer to separate Laying with virtually 100% confidence.

---

## 6.4 Attention Distribution: Moving vs. Still Activities

**Question:** Is the attention more peaked for moving or for still activities? Explain what that means about where the useful information is in the window.

**Answer:**  
- **Empirical Peak Values ($\text{mean of } \max_t \alpha_t$):**
  - **Moving Activities:**
    - `WALKING`: **0.2978**
    - `WALKING_UPSTAIRS`: **0.2364**
    - `WALKING_DOWNSTAIRS`: **0.2784**
  - **Still Activities:**
    - `SITTING`: **0.0144**
    - `STANDING`: **0.0184**
    - `LAYING`: **0.0106**
  *(For reference, completely uniform attention across 128 time steps is $1/128 \approx 0.0078$).*

- **Interpretation:**  
  Attention is **dramatically more peaked for moving activities** (peaking around 24%–30% on isolated steps) compared to still activities (diffuse across time, staying near 1%–1.8%).  
  - For **dynamic activities**, information is concentrated in transient biomechanical events: heel-strike impacts, push-off phases, and peak knee flexions that produce sharp acceleration/gyroscope extrema. The attention mechanism selectively focuses its temporal weight on these discriminative moments.
  - For **static activities**, the physiological signal is time-invariant and stationary. Every time step carries essentially identical information (the static orientation of gravity). Consequently, the attention weights remain flat and evenly distributed across the entire 2.56-second sequence.

---

## 6.5 Bidirectional Recurrence in Offline vs. Online Tasks

**Question:** A bidirectional LSTM reads the window backwards as well as forwards. Why is that allowed here, and in what kind of task would it not be allowed?

**Answer:**  
- **Why Allowed Here:**  
  Our application operates in an **offline windowed classification** paradigm. A 2.56-second buffer of sensor samples (128 time steps) is captured in its entirety before being passed to the network. Because the complete temporal window is already recorded, the backward pass has full causal access to subsequent samples within that window.
- **When It Would NOT Be Allowed:**  
  Bidirectional processing is strictly disallowed in **real-time, low-latency streaming applications** requiring step-by-step causal prediction at time $t$ with zero buffer delay:
  1. *Immediate fall detection / airbag deployment:* Waiting 2.56 seconds for the window to finish before classifying would exceed the injury threshold.
  2. *Real-time step counters and interactive VR locomotion:* Delay would introduce perceptible tracking latency.
  3. *Autoregressive generation / next-step forecasting:* Future steps $t+1, \dots$ do not yet exist.

---

## 6.6 LSTM Gating vs. Plain RNN & Gradient Clipping

**Question:** You implemented an RNN cell but trained an LSTM. Using the update $c_t = f_t \odot c_{t-1} + i_t \odot g_t$, explain why an LSTM keeps information over 128 steps better than a plain RNN. What problem does gradient clipping guard against?

**Answer:**  
1. **Information Retention Over 128 Steps:**  
   In a vanilla RNN, the hidden state update is $h_t = \tanh(W x_t + U h_{t-1} + b)$. During Backpropagation Through Time (BPTT) across 128 steps, the gradient of the loss with respect to early hidden states $h_1$ involves the repeated product:
   $$\frac{\partial h_T}{\partial h_1} = \prod_{k=2}^T U^T \text{diag}(1 - \tanh^2(\dots))$$
   Because $\tanh'(z) \le 1$, continuous multiplication across 128 steps causes gradients to decay exponentially toward zero whenever eigenvalues of $U$ are small (the **vanishing gradient problem**), preventing the RNN from capturing long-range dependencies.  
   
   In contrast, the LSTM cell maintains an additive internal memory cell $c_t$:
   $$c_t = f_t \odot c_{t-1} + i_t \odot g_t$$
   The derivative of the cell state with respect to the previous cell state is simply:
   $$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$
   When the forget gate $f_t \approx 1$, the error gradient flows backwards through time additively with minimal attenuation—an "error carousel." This linear highway allows gradients and memory signals to persist intact across 128 steps.

2. **Role of Gradient Clipping:**  
   Gradient clipping (`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`) guards against **exploding gradients**. In recurrent models, if weights are large, repeated multiplication can cause backpropagated gradients to grow exponentially ($> 10^3$). A single massive gradient step would destabilize the optimizer, catapulting weights into regions of high loss or causing numerical overflow (`NaN`). Clipping scales down oversized gradients while preserving their directional vector, ensuring stable gradient descent.
