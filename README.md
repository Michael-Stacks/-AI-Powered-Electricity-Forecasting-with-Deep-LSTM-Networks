🧠 AI Architecture & Training

Deep Neural Network Design
Core Model: Multi-layered LSTM with 128 hidden units and 2 recurrent layers.

Loss Function: Mean Squared Error (MSE) for regression.

Optimizer: Adam algorithm with learning rate scheduling.

Training Process
Data Preparation:

Sliding time windows (7-day sequences) with overlap to preserve temporal context.

Train/Validation/Test split (70%/15%/15%).

GPU Acceleration:

Leverages CUDA for faster training (if available).

Epochs:

50 epochs with batch size 32, achieving loss convergence.



📈 Key AI Metrics

Training Loss: Dropped from 0.0014 → 0.0012 (optimal learning).

Validation Loss: Stabilized at ~0.0116, highlighting generalization potential.

Speed: 130–140 iterations/sec – efficient for deep networks.




🎯 Why This Stands Out

Deep Learning Focus: Explores advanced AI techniques for time-series forecasting.

Scalability: Easily extendable to larger datasets or hybrid architectures (e.g., CNN-LSTM).

Real-World Impact: Directly applicable to energy companies for grid optimization.




🌟 Future AI Enhancements

Attention Mechanisms: Improve focus on critical time steps.

Transfer Learning: Pre-train on similar datasets.

Hyperparameter Tuning: Bayesian optimization for better convergence.




![main py - 8INF406---LSTM-main - Visual Studio Code 2025-03-05 16_15_41](https://github.com/user-attachments/assets/89e581b0-6ea1-4284-a07f-cb93f411bdbc)

![main py - 8INF406---LSTM-main - Visual Studio Code 2025-03-05 16_16_09](https://github.com/user-attachments/assets/0e376130-a64c-4557-8fa2-caf64cbcca64)

![loss_curves](https://github.com/user-attachments/assets/c36dd387-0f3c-4eb5-833d-be4a4b7c8d1c)

![production](https://github.com/user-attachments/assets/f1ba3d53-2adc-4374-bf54-eb8e294fa222)

![consumption](https://github.com/user-attachments/assets/01fa9169-548d-410f-b68a-ca54cef89618)





