# Explainable AI for Credit Risk Assessment

A comparative machine learning study investigating the **reliability, reproducibility and fairness of Explainable AI (XAI) methods for credit risk assessment**.

The project compares multiple predictive models and explanation techniques across two established credit risk datasets, with particular attention given to whether explanations remain consistent across modelling approaches, software frameworks and random initialisations.

## Project Overview

Machine learning models are increasingly used to support financial decision-making, including credit risk assessment. While complex models can provide strong predictive performance, understanding **why a model makes a prediction** is particularly important in high-stakes applications.

This project investigates how reliable commonly used Explainable AI methods are when applied to different credit risk models.

The analysis focuses on three areas:

1. **Agreement** – Do different XAI methods identify similar important features?
2. **Reproducibility** – Are explanations stable when models are retrained using different random seeds and machine learning frameworks?
3. **Fairness** – How do model explanations relate to demographic fairness measures?

## Research Questions

The project investigates the following broad questions:

* How consistently do different Explainable AI methods explain credit risk models?
* How reproducible are neural network explanations across TensorFlow and PyTorch implementations?
* What relationship exists between model explanations and demographic fairness?

## Datasets

Two established credit risk datasets were used to assess whether findings generalised across datasets with different characteristics.

### Taiwan Default of Credit Card Clients

Approximately **30,000 customer records** containing demographic information, credit limits, repayment history and previous payment behaviour.

**Target:** Whether a customer defaults on their credit payment.

### German Statlog Credit

A smaller dataset containing **1,000 credit applicants**, with financial, demographic and credit-related characteristics.

**Target:** Credit risk classification.

The original datasets are not redistributed through this repository. Please refer to their respective source repositories for access.

## Machine Learning Models

Four model implementations were evaluated:

* **Logistic Regression**
* **XGBoost**
* **TensorFlow Feed-Forward Neural Network**
* **PyTorch Feed-Forward Neural Network**

The TensorFlow and PyTorch neural networks were designed as matched architectures to enable investigation of whether implementation framework affected model explanations.

## Explainable AI Methods

Three widely used explanation approaches were investigated:

### SHAP

SHapley Additive exPlanations estimate feature contributions using concepts derived from cooperative game theory.

### LIME

Local Interpretable Model-Agnostic Explanations approximate model behaviour around individual predictions using interpretable local surrogate models.

### Integrated Gradients

Integrated Gradients is a gradient-based attribution method used to explain neural network predictions by measuring feature contributions relative to a baseline.

## Evaluation

### Predictive Performance

Model discrimination was assessed using cross-validated **Area Under the Receiver Operating Characteristic Curve (ROC-AUC)**.

### Explanation Agreement

Agreement between feature importance rankings produced by different XAI methods was investigated using techniques including:

* Kendall's coefficient of concordance
* Spearman rank correlation

### Reproducibility

The neural network experiments were repeated across **10 random seeds**.

Attribution results from matched TensorFlow and PyTorch models were compared using:

* Paired Wilcoxon signed-rank tests
* Bonferroni correction for multiple comparisons

### Fairness

Demographic fairness was investigated using:

* Disparate impact
* Four-fifths rule
* Bootstrap confidence intervals

This allowed model behaviour and feature attribution results to be considered alongside group-level fairness measures.

## Key Findings

The experiments showed that explanation consistency depends on the combination of **dataset, predictive model and XAI technique**.

Across the experiments:

* XAI methods showed moderate to strong levels of agreement in many model and dataset combinations.
* Explanation rankings were not perfectly interchangeable between XAI techniques.
* Matched TensorFlow and PyTorch neural networks produced broadly comparable attribution patterns, although small differences were observed.
* Repeating neural network experiments across multiple random seeds demonstrated the importance of considering attribution stability rather than relying on a single model run.
* Fairness analysis provided an additional perspective on how model predictions and explanations interact with demographic characteristics.

These findings highlight the importance of evaluating the **reliability of explanations themselves**, rather than assuming that the output of an XAI method is automatically stable or reproducible.

## Repository Structure

```text
.
├── notebooks/
│   ├── Dissertation_Taiwan.ipynb
│   ├── Dissertation_German.ipynb
│   └── Experimental_Design_Figure.ipynb
│
├── r/
│   └── Statistical_Analysis_and_Figures.Rmd
│
├── data/
│   ├── german_cv_auc.csv
│   ├── german_rq2_shap.csv
│   ├── german_rq2_training.csv
│   ├── german_test_predictions.csv
│   ├── german_xai_importance.csv
│   ├── taiwan_cv_auc.csv
│   ├── taiwan_rq2_shap.csv
│   ├── taiwan_rq2_training.csv
│   ├── taiwan_test_predictions.csv
│   └── taiwan_xai_importance.csv
│
└── README.md
```

The CSV files contained in `data/` are generated analysis outputs used for the statistical analysis and visualisation stages of the project rather than copies of the original credit datasets.

## Technologies

### Python

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* TensorFlow
* PyTorch
* SHAP
* LIME
* Matplotlib
* Seaborn

### R

* R
* R Markdown
* Statistical hypothesis testing
* Data visualisation

### Methods

* Binary classification
* Cross-validation
* Neural networks
* Explainable AI
* Feature attribution
* Statistical hypothesis testing
* Bootstrap confidence intervals
* Model reproducibility analysis
* Algorithmic fairness analysis

## Workflow

The project followed an experimental workflow consisting of:

**Data preprocessing → Model training → Predictive evaluation → XAI generation → Explanation comparison → Reproducibility testing → Fairness analysis**

Separate experiments were performed on the Taiwan and German credit datasets before results were combined for statistical analysis and interpretation.

## Why This Project Matters

Explainability is increasingly important when machine learning is used in high-stakes decision-making.

However, producing an explanation is not sufficient by itself. Explanations should also be assessed for **agreement, stability and reproducibility**.

This project demonstrates a framework for evaluating not only predictive model performance, but also the reliability of the explanations used to interpret those predictions.

## Future Development

Potential extensions include:

* evaluating additional credit risk datasets;
* comparing further XAI techniques;
* investigating additional fairness metrics;
* assessing explanation stability under distribution shift;
* developing an interactive interface for exploring local and global model explanations.

## Author

**Gabriella Davis**

MSc Data Science

GitHub: `gabriella-davis0505`

---

*This repository contains a portfolio version of work completed as part of an MSc Data Science project. The repository focuses on the technical implementation, experimental methodology and results.*
