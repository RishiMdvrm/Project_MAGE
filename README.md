# MAGE Model Reproduction

## Introduction

This repository contains the implementation of experiments using DistilBERT and Longformer models for text classification and detection tasks. The directory includes training scripts, main program files, and setup requirements for reproducibility. Both models are configured to handle large-scale datasets and can be fine-tuned for text-based classification tasks.

---

## 📝 Dataset

You can find the dataset in the folder Data. It comprises of three file:
 - train.csv
 - valid.csv
 - test.csv
The data in this files comprises of both human and AI generated text gathered from various domains and language models.

---

## Setup

### Clone the repo
```bash
git clone https://github.com/RishiMdvrm/Project_MAGE.git
cd Project_MAGE
```

### Python Environment

For deploying the Longformer detector or training your own detector using our data, create a virtual environment simply run the following command:

```shell
pip install -r requirements.txt
```

### Model Access

To run the models, you can go to the training folder:
Training
├── DistilBERT
│   ├── main.py         # Main script for training, evaluation, and prediction using DistilBERT
│   ├── train.sh        # Shell script to configure and execute training for DistilBERT
├── longformer
│   ├── main.py         # Main script for training, evaluation, and prediction using Longformer
│   ├── train.sh        # Shell script to configure and execute training for Longformer
├── requirements.txt    # Python dependencies required to run the scripts

You can choose either DistilBERT or longformer and run the train.sh file using an IDE or the below script
> **Note:** Change the directory names in the `train.sh` file as per your local setup.

```shell
bash train.sh
```

---

## Output
The results of training and evaluation will be saved in the output_dir specified in train.sh. Each run generates:

- Model checkpoints.
- Training and evaluation logs.
- Metrics such as accuracy, precision, recall, F1-score, and AUROC.
- Visualizations (e.g., training loss plots, confusion matrix).

---

## Evaluation Metrics

Both models are evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **AUROC** 

Metrics are logged and stored in the specified output directory.

---

## Notes

- Longformer is configured for handling sequences up to 2048 tokens, making it suitable for long-text tasks.
- DistilBERT, a lightweight alternative, processes sequences up to 256 tokens efficiently.
- Ensure adequate GPU resources for Longformer training, as it is computationally intensive.

---

## Acknowledgements

This work is based on the [MAGE GitHub Repository](https://github.com/yafuly/MAGE/tree/main). The codebase was adapted for specific experiments in text classification and detection tasks.

For any questions or issues, feel free to reach out to me at rmadha4@uic.edu. Happy experimenting!