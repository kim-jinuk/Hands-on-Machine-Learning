# Hands-On Machine Learning – Study Repository

이 레포지토리는 \*\*Aurélien Géron, \*\****Hands‑On Machine Learning with Scikit‑Learn, Keras & TensorFlow*** 책을 챕터별로 학습하며 작성한 예제 코드와 연습 문제 풀이를 정리합니다.

---

## 폴더 구조

```text
.
├── ch01_the_machine_learning_landscape/
│   ├── notebook.ipynb
│   └── solutions/
├── ch02_end_to_end_ml_project/
├── ...
├── ch20_autoencoders_and_gans/
├── environment.yml   # 선택 사항
└── README.md
```

각 챕터 디렉터리 이름은 `chXX_<title>/` 형식을 사용합니다.

---

## 환경 설정

```bash
# Conda 사용 예시
conda env create -f environment.yml
conda activate homl

# JupyterLab 실행
jupyter lab
```

> `environment.yml` 은 CPU 버전 기준입니다. GPU TensorFlow가 필요하면 해당 항목을 교체하세요.

---

## 목차

| Chapter | Title                              | Code Dir                                     | Exercise List |
| ------- | ---------------------------------- | -------------------------------------------- | ------------- |
| 1       | The Machine Learning Landscape     | [ch01](ch01_the_machine_learning_landscape/) | [Jump](#ch1)  |
| 2       | End‑to‑End ML Project              | [ch02](ch02_end_to_end_ml_project/)          | [Jump](#ch2)  |
| 3       | Classification                     | [ch03](ch03_classification/)                 | [Jump](#ch3)  |
| 4       | Training Models                    | [ch04](ch04_training_models/)                | [Jump](#ch4)  |
| 5       | Support Vector Machines            | [ch05](ch05_support_vector_machines/)        | [Jump](#ch5)  |
| 6       | Decision Trees                     | [ch06](ch06_decision_trees/)                 | [Jump](#ch6)  |
| 7       | Ensemble Learning & Random Forests | [ch07](ch07_ensemble_learning/)              | [Jump](#ch7)  |
| 8       | Dimensionality Reduction           | [ch08](ch08_dimensionality_reduction/)       | [Jump](#ch8)  |
| 9       | Unsupervised Learning              | [ch09](ch09_unsupervised_learning/)          | [Jump](#ch9)  |
| 10      | Artificial Neural Networks         | [ch10](ch10_ann/)                            | [Jump](#ch10) |
| 11      | Deep Computer Vision with CNNs     | [ch11](ch11_cnn/)                            | [Jump](#ch11) |
| 12      | RNNs for Sequence Modeling         | [ch12](ch12_rnn/)                            | [Jump](#ch12) |
| 13      | Attention Mechanisms               | [ch13](ch13_attention/)                      | [Jump](#ch13) |
| 14      | Transformers & NLP                 | [ch14](ch14_transformers/)                   | [Jump](#ch14) |
| 15      | Training & Serving TF Models       | [ch15](ch15_tf_serving/)                     | [Jump](#ch15) |
| 16      | TFX Pipeline                       | [ch16](ch16_tfx/)                            | [Jump](#ch16) |
| 17      | Model Deployment                   | [ch17](ch17_deployment/)                     | [Jump](#ch17) |
| 18      | Reinforcement Learning             | [ch18](ch18_rl/)                             | [Jump](#ch18) |
| 19      | Generative Deep Learning           | [ch19](ch19_generative_dl/)                  | [Jump](#ch19) |
| 20      | Autoencoders & GANs                | [ch20](ch20_autoencoders_and_gans/)          | [Jump](#ch20) |

---

## 연습 문제 체크리스트

> 풀이를 완료하면 `[ ]` 를 `[x]` 로 변경해 주세요.

### Chapter 1 – The Machine Learning Landscape

-

### Chapter 2 – End‑to‑End ML Project

-

### Chapter 3 – Classification

-

### Chapter 4 – Training Models

-

### Chapter 5 – Support Vector Machines

-

---

## 기여 방법

1. 레포지토리를 포크한 뒤 새 브랜치(`feature/ch03_solution`)를 만듭니다.
2. 노트북이나 스크립트를 추가하고 README 체크박스를 업데이트합니다.
3. Pull Request를 생성하면서 변경 사항을 간단히 설명합니다.

PR은 언제든 환영입니다!

---

## 라이선스

- 코드와 노트북: MIT
- 책 내용 요약·인용: © O’Reilly Media – 공정 사용 범위 내 인용

---

Happy coding and committing! 🚀
