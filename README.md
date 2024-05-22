Here's the plain text you can copy and insert into your GitHub README:

---

# Semantic Equivalence Evaluation Project

## Introduction
This project evaluates the semantic equivalence of question pairs using three different models:
- **GloVe Embeddings with Cosine Similarity**
- **Shallow Neural Network**
- **BERT Hidden States with Cosine Similarity**

## Features
- **GloVe Embeddings**: Uses pre-trained GloVe embeddings to represent questions as vectors and measures their similarity using cosine similarity. A hyperparameter search is performed to find optimal cutoff thresholds.
- **Shallow Neural Network**: Prompts the user for the number of question pairs to evaluate, then takes input for the questions to determine semantic equivalence.
- **BERT Hidden States**: Utilizes hidden states from BERT to represent questions and measures their similarity using cosine similarity.

## System Requirements
- **RAM**: At least 93GB

## Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/samagdur/projectdd24
    ```
2. **Create a virtual environment and activate it**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Running the Program**:
    ```bash
    python LE_proj.py
    ```
2. **User Interaction**:
    - For the **Shallow Neural Network** model, the program will prompt you to input the number of question pairs you want to evaluate.
    - You will then be asked to type each pair of questions.
    - The program will output whether the sentences are semantically equivalent or not.

## Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

## License
This project is licensed under the MIT License.

## Acknowledgements
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

## Contact
For any questions, please open an issue on GitHub or contact the project maintainer at [sam4@kth.se](mailto:sam4@kth.se).

---
