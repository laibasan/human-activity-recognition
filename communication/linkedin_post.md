# 📱 Decoding Movement: Building a BiLSTM with Attention from Scratch (and Wrapping It in Van Gogh's Starry Night) ✨

Can your smartphone recognize whether you're walking, climbing stairs, or sitting down—purely from raw accelerometer and gyroscope signals?

For our latest Applied Machine Learning project, my teammate and I built a complete **Human Activity Recognition (HAR)** pipeline from first principles in **PyTorch**. The twist? **No black-box `nn.LSTM` or `nn.MultiheadAttention` allowed.** Every gate, cell state, bidirectional pass, and attention weight was written from mathematical equations.

Here is what we learned and built:

---

### 🔬 The Engineering & Architecture
- **Raw Sensor Data:** 128 time-step windows (2.56 s @ 50 Hz) across 9 inertial channels (body acc, gyroscope, total acc) from the benchmark UCI HAR dataset.
- **Custom BiLSTM Cell:** Built gate equations ($i_t, f_t, g_t, o_t, c_t, h_t$) from scratch with unified weight packing (37,888 params).
- **Additive Attention:** Implemented Bahdanau-style temporal pooling ($e_t = v^T \tanh(W_a h_t + b_a)$) to interpret which moments in the 2.56-second window dictate the prediction (8,320 params).
- **Total Model Size:** Just **46,982 parameters**!

### 📊 Results & Key Findings
- **91.45% Test Accuracy** across 2,947 held-out test windows.
- **100.00% Accuracy on Laying:** When horizontal, Earth's $1g$ gravity shifts 90° onto the orthogonal `total_acc_x` axis—an unmistakable physical signature.
- **The Sitting vs. Standing Dilemma:** Confused most often because waist-worn static postures produce nearly identical tilt angles and zero dynamic acceleration without transition context.
- **Where Does the Network Look?** Attention peaks sharply (~30% weight) during sudden foot-strike impacts in walking activities, but diffuses evenly (~1%) across static sitting/standing where no single moment carries unique information.

---

### 🎨 The Interface: Van Gogh's Starry Night
We exported the trained weights to JSON and built a standalone JavaScript inference engine (`infer.js`) running entirely in the browser. Wrapped in an animated, interactive canvas inspired by Vincent van Gogh's *The Starry Night* (1889), complete with swirling nebulae, glowing celestial halos, live signal waveforms, and real-time attention heatmaps!

---

🔗 **Explore the Project:**
- 📖 **Full Medium Article:** https://medium.com/@laibamazhersardar/listening-to-the-rhythm-of-movement-building-a-bilstm-with-attention-from-scratch-for-human-activity-recognition
- 🌐 **Live Interactive Web Demo:** https://laibasan.github.io/human-activity-recognition/
- 💻 **GitHub Repository:** https://github.com/laibasan/human-activity-recognition

Huge thanks to our instructors and peers! Would love your thoughts and feedback in the comments. 👇

#MachineLearning #DeepLearning #PyTorch #DataScience #ArtificialIntelligence #BiLSTM #AttentionMechanism #HumanActivityRecognition #WebDevelopment #VanGogh #OpenSource
