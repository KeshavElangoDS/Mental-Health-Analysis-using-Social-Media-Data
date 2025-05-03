# Model Card

**Model:** prajjwal1/bert-tiny

**Model Path:** onnx/bert_tiny_model_optimized.onnx

**Training Parameters:**
* batch_size: 64
* test_batch_size: 64
* epochs: 5
* learning rate: 2e-5
* patience : 2
* no_cuda: true
* no_mps: false
* use_profiler: False
* optimizer_type: AdamW
* interval: step
* loss_function: CrossEntropyLoss

**Model accuracy:**

- *Training Accuracy*: `0.8992`
- *Validation Accuracy*: `0.9589`
- *AUC-ROC (OvR)*: `0.9979`

**Model training**:
* The model was trained using a batch size of 64, for 5 epochs with a learning rate of 2e-5.
* The CrossEntropyLoss loss function was used for calculating the loss.
* We have a parameter called patience which determines when to stop the training provided the model does not improve after certain epochs.
* The number of epochs is determiend by the parameter patience.
* Uses GPU/CUDA if available else uses MPS otherwise uses CPU. Here MPS is used with precision of 16.
* As the model used is bertbeing a transformer model for text classification, PyTorch Lightning is used.
* Optimizers aim to speed up training and improve the model's performance while preventing overfitting.
* **AdamW** is an optimizer adapts the learning rate for each parameter.

**Limitations:**

* Overfitting on small datasets despite AdamW's regularization.
* Computational overhead from AdamW's momentum terms.
* BERT models are computationally expensive requiring GPU or Apple MPS backend for reasonable training time.
* High memory usage can cause out-of-memory errors, especially for long sequences or large batch sizes.
