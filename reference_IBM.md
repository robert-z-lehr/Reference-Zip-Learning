# Reference Sheet:
## Algorithms - Problem Categories - Terminology/Jargon
---
### Common problems categories and algorithms in data science along with their respective terminology and jargon.

- *Source:* ChatGPT4

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

### Core useful Python concepts and techniques:
Certainly, here are key learning takeaways based on those questions and other useful considerations:

#### `range()` function:
1. `range()` can be used with one, two, or three arguments: `range(stop)`, `range(start, stop)`, and `range(start, stop, step)`.
2. The `stop` value is exclusive, and the default `start` value is 0.

#### While Loop and `break`:
3. You can exit a `while` loop prematurely using the `break` statement.
4. Always make sure the condition in `while` eventually evaluates to `False` to prevent infinite loops.

#### Looping in Reverse:
5. You can loop through a sequence in reverse using `reversed()` or setting the `step` argument in `range()` to `-1`.

#### Initialize Variable Based on Iteration:
6. Variables can be dynamically named, though this is not recommended. Use data structures like dictionaries for this purpose instead.

#### `enumerate()`:
7. `enumerate()` returns an iterator that produces tuples containing indices and value from the iterable.
8. It's useful when you need to track both the index and the value in the loop.

#### Avoiding Index Out-of-Bounds:
9. Always check that the index is within the range of the list's size: `0 <= index < len(list)`.
10. IndexError is typically raised when you try to access an index that is out of the range of valid indices for a list, string, or other indexed data structure.

#### Error Identification:
11. `IndexError` is raised when you try to access an index that is out of bounds. Make sure to handle this appropriately in your code, usually with a conditional statement to check the index range or by using try-except blocks.

#### Output Formatting:
12. You can control the end character in `print()` by setting the `end` parameter (default is a newline).
13. Floating-point numbers can be formatted to a fixed number of decimal places using format specifiers like `:.6f`.

#### Namespaces:
14. A namespace is a container that holds a set of identifiers (variables, functions, classes, etc.) and allows the disambiguation of their usages.

Certainly, here are 20 more key Python concepts and techniques that you may find useful in your programming journey:

#### List Comprehensions:
15. List comprehensions provide a concise way to create lists: `[x**2 for x in range(10)]`.

#### Lambda Functions:
16. Lambda functions are small anonymous functions that can take any number of arguments but can only have one expression.
```
square = lambda x: x * x
print(square(5))  # Output: 25
```
- In this example, lambda x: x * x defines a function that squares the input. We assign this lambda function to the variable square and then use it to square 5.

#### Tuples and Immutability:
17. Tuples are similar to lists, but they are immutable. Once a tuple is created, you can't change its values.
```
my_tuple = (1, 2, 3)
# my_tuple[0] = 4  # This would raise a TypeError because tuples are immutable
print(my_tuple[0])  # Output: 1
```
In this example, my_tuple is a tuple containing three integers. Attempting to change an element of the tuple (as commented out) would raise a TypeError.

#### Dictionary Operations:
18. Dictionaries are mutable, and key-value pairs can be added or removed dynamically using `dict[key] = value` and `del dict[key]`.

#### Slicing:
19. Slicing is a feature that allows you to obtain a sublist, substring, or subarray from a list, string, or array. The syntax is `sequence[start:stop:step]`.

#### Function Arguments:
20. Python functions support positional, keyword, and default arguments. You can also use `*args` and `**kwargs` to pass variable-length argument lists.

#### Global and Local Variables:
21. Variables declared inside a function are local to that function. You can use the `global` keyword to modify a global variable inside a function.

#### Generators:
22. Generators are iterators that yield items lazily, which can save memory when dealing with large data sets.

#### Exception Handling:
23. Use `try`, `except`, `finally` blocks to catch and handle exceptions in Python.

#### `map()` and `filter()`:
24. `map()` and `filter()` are higher-order functions that apply a function to each element in an iterable (map) or filter elements based on a condition (filter).

#### Modules and Packages:
25. Modules are `.py` files that contain Python code. Packages are collections of modules. Use `import` to include them in your script.

#### `with` Statement:
26. Use the `with` statement to simplify resource management like file I/O operations.

#### String Manipulation:
27. Python strings have built-in methods for common operations like `lower()`, `upper()`, `split()`, and `join()`.

#### File Operations:
28. Python has built-in functions for file reading (`open()`, `read()`) and writing (`write()`).

#### Decorators:
29. Decorators are higher-order functions that allow you to add functionality to an existing function by passing the existing function to a decorator.

#### Regular Expressions:
30. The `re` module provides regular expression matching operations.

#### Type Annotations:
31. Python 3.5+ supports optional type annotations for documenting the expected data types of function arguments and return values.

#### The `pass` Statement:
32. The `pass` statement is a null operation that serves as a placeholder where syntactically some code is required.

#### Assertion:
33. The `assert` statement can be used to insert debugging checks. It raises an `AssertionError` if a specified condition is not true.

#### Context Managers:
34. Context managers ensure resources are properly and automatically managed using `__enter__` and `__exit__` methods, commonly used in file operations and network connections.

---

## Python datatypes and data structures,  rigorously categorized:

### Built-in Python Data Types:
1. `int` - Integer numbers
2. `float` - Floating-point numbers
3. `bool` - Boolean (`True` or `False`)
4. `str` - String
5. `bytes` - Byte literals
6. `bytearray` - Mutable sequence of bytes
7. `NoneType` - Type of `None`

### Built-in Python Data Structures:
1. `list` - Ordered, mutable sequence
2. `tuple` - Ordered, immutable sequence
3. `set` - Unordered collection of unique elements
4. `dict` - Unordered collection of key-value pairs
5. `frozenset` - Immutable version of a set

#### Creating a Custom Data Type in Python:
In Python, you can create a custom data type using classes. For example, you could define a data type for representing a point in a two-dimensional space.

```python
class Point2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y
```

Usage:

```python
p = Point2D(3, 4)
```

#### Creating a Custom Data Structure in Python:
You can also create custom data structures using classes. Here's a basic example where we implement a simple stack.

```python
class Stack:
    def __init__(self):
        self.items = []
        
    def push(self, item):
        self.items.append(item)
        
    def pop(self):
        return self.items.pop()
        
    def is_empty(self):
        return len(self.items) == 0
```

Usage:

```python
s = Stack()
s.push(1)
s.push(2)
print(s.pop())  # Output: 2
```

The `Point2D` class serves as a custom data type to represent a 2D point, while the `Stack` class is a custom data structure for managing items in a last-in, first-out (LIFO) manner.