# Reference Sheet:
## Algorithms - Problem Categories - Terminology/Jargon
---
### Common problems categories and algorithms in data science along with their respective terminology and jargon. *Source:* ChatGPT4

---

#### Common Data Science Problem Categories
| Problem Category | Description |
|------------------|-------------|
| [**Classification**](https://en.wikipedia.org/wiki/Statistical_classification) | Identifying to which of a set of categories a new observation belongs. |
| [**Regression**](https://en.wikipedia.org/wiki/Regression_analysis) | Predicting a continuous value based on input variables. |
| [**Clustering**](https://en.wikipedia.org/wiki/Cluster_analysis) | Grouping a set of objects so that objects in the same group are more similar to each other than to those in other groups. |
| [**Dimensionality Reduction**](https://en.wikipedia.org/wiki/Dimensionality_reduction) | Reducing the number of random variables under consideration. |
| [**Time Series Analysis**](https://en.wikipedia.org/wiki/Time_series) | Analyzing time-ordered data points. |
| [**Anomaly Detection**](https://en.wikipedia.org/wiki/Anomaly_detection) | Identifying abnormal or rare items in a data set. |
| [**Association Rule Mining**](https://en.wikipedia.org/wiki/Association_rule_learning) | Discovering interesting relations between variables in large databases. |
| [**Reinforcement Learning**](https://en.wikipedia.org/wiki/Reinforcement_learning) | Training models to make a sequence of decisions. |
| [**Natural Language Processing (NLP)**](https://en.wikipedia.org/wiki/Natural_language_processing) | Enabling computers to understand, interpret and produce human language. |
| [**Recommendation Systems**](https://en.wikipedia.org/wiki/Recommender_system) | Providing personalized recommendations to users.|
| [**Ensemble Learning**](https://www.sciencedirect.com/science/article/pii/S1877050918312463) | Improving performance by training multiple learners to solve the same problem.|
| [**Neural Networks**](https://en.wikipedia.org/wiki/Artificial_neural_network) | Recognizing patterns and interpreting data through machine perception, labeling, and clustering.|

--- 

#### Common Data Science Algorithms
| Algorithm | Description |
|-----------|-------------|
| [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression) | Assumes a linear relationship between inputs and the target variable. |
| [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression) | Used for binary outcomes. |
| [Decision Trees](https://en.wikipedia.org/wiki/Decision_tree) | A flowchart-like tree structure for decision-making. |
| [Random Forest](https://en.wikipedia.org/wiki/Random_forest) | Averages multiple decision trees for more accurate predictions. |
| [Support Vector Machines (SVM)](https://en.wikipedia.org/wiki/Support_vector_machine) | Used for classification and regression tasks. |
| [K-Nearest Neighbors (KNN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) | Classifies a data point based on how its neighbors are classified. |
| [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) | A probabilistic classifier based on Bayes' theorem. |
| [Principal Component Analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis) | Transforms original variables into a new set of orthogonal | variables. |
| [K-Means Clustering](https://en.wikipedia.org/wiki/K-means_clustering) | Divides a dataset into k distinct, non-overlapping clusters. |
| [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) | Combines multiple weak learning models to create a strong predictive model. |

---

#### Common Data Science Jargon and Terminology

| Term | Definition |
|------|------------|
| [Bias](https://en.wikipedia.org/wiki/Bias_(statistics)) | The error introduced by approximating a real-world problem, often seen as the difference between the prediction and the actual outcome. |
| [Variance](https://en.wikipedia.org/wiki/Variance) | The spread of a data distribution, or how far individual points differ from the mean. In machine learning, high variance often suggests overfitting. |
| [Overfitting](https://en.wikipedia.org/wiki/Overfitting) | A model that has learned the training data too well, including its noise and outliers, thereby performing poorly on new data. |
| [Underfitting](https://en.wikipedia.org/wiki/Underfitting) | A model that has not learned enough from the training data, resulting in poor performance on both the training and new data. |
| [Cross-Validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) | A technique for assessing the effectiveness of a model by partitioning the original training data set into a training set and a validation set. |
| [F1 Score](https://en.wikipedia.org/wiki/F1_score) | The harmonic mean of precision and recall, used as a single metric for model evaluation. |
| [Precision](https://en.wikipedia.org/wiki/Precision_and_recall) | The number of true positives divided by the number of true positives and false positives. |
| [Recall](https://en.wikipedia.org/wiki/Precision_and_recall) | The number of true positives divided by the number of true positives and false negatives. |
| [AUC-ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) | The Area Under the Receiver Operating Characteristic Curve, a performance measurement for classification problems. |
| [Regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics)) | Techniques used to prevent overfitting by adding additional information or constraints to a model. |
| [Batch Processing](https://en.wikipedia.org/wiki/Batch_processing) | Processing high volumes of data where a group of transactions is collected over a period of time. |
| [Feature Engineering](https://en.wikipedia.org/wiki/Feature_engineering) | The practice of selecting and transforming variables when creating a predictive model. |
| [Hyperparameter](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) | Parameters of the learning process, as opposed to those of the model itself. |
| [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix) | A table used to describe the performance of a classification model. |
| [Imputation](https://en.wikipedia.org/wiki/Imputation_(statistics)) | The process of replacing missing data with substituted values. |

---

#### Data Science Problem Categories, Algorithms, and Example Cases

| Problem Category | Common Algorithms | Example Cases |
|------------------|-------------------|---------------|
| Classification   | - Logistic Regression<br>- Random Forest<br>- SVM | - Email spam filtering<br>- Customer churn prediction<br>- Medical diagnosis |
| Regression       | - Linear Regression<br>- Ridge Regression<br>- Lasso Regression | - House price prediction<br>- Stock market forecasting<br>- Sales estimation |
| Clustering       | - K-Means<br>- Hierarchical Clustering<br>- DBSCAN | - Customer segmentation<br>- Social network analysis<br>- Image segmentation |
| Anomaly Detection| - Isolation Forest<br>- One-Class SVM<br>- k-NN | - Fraud detection<br>- Network intrusion detection<br>- Outlier detection in manufacturing |
| Dimensionality Reduction | - PCA<br>- t-SNE<br>- LDA | - Feature selection in high-dimensional data<br>- Data visualization<br>- Noise reduction |
| Time Series Analysis | - ARIMA<br>- LSTM<br>- Prophet | - Weather forecasting<br>- Financial market analysis<br>- Energy consumption prediction |
| Natural Language Processing | - Naive Bayes<br>- LSTM<br>- BERT | - Sentiment analysis<br>- Machine translation<br>- Text summarization |
| Recommender Systems | - Collaborative Filtering<br>- Content-Based Filtering<br>- Matrix Factorization | - Product recommendation in e-commerce<br>- Movie recommendation<br>- News article recommendation |
| Reinforcement Learning | - Q-Learning<br>- Deep Q Network<br>- Policy Gradients | - Game playing<br>- Robotics<br>- Resource allocation |
| Ensemble Learning | - Bagging<br>- Boosting<br>- Stacking | - Credit scoring<br>- Ensemble classifiers in healthcare<br>- Improved image recognition |

---