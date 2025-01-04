# ML Basics Chapter 2 - Three Components of ML

# Introduction

- ML methods fit a model to data by minimizing a loss function.
- Main components: model, loss, and data.
    - **Data**: collection of individual data points that are characterized by features.
    - **Model/Hypothesis Space**: consists of computation feasible hypothesis maps from a feature space to label space.
    - **Loss Function**: measures the quality of a hypothesis map.

# 2.1 The Data

- Data can be quantitative or qualitative.
- **Sample size**: the number of data points within a dataset.
    - We will assign m to be the maximum sample size.
    - larger m ⇏ better
        - issues with computational resources
- **Features**: characteristics of individual data points.
    - We will assign n to the number of features of each data point.
- The behaviour of ML methods depend on the m - n ratio.
    - Typically improve with increasing m/n.
    - Rule of thumb: use datasets for which m/n > 1
- **Euclidean space**:  **ℝⁿ**
    - Reflects the structure of the data: dimension == number of features.

### 2.1.2 Labels

- **Label**: the higher-level fact or quantity of interest that is associated with a data point, i.e. the properties.
    - For a data point: If the features (x) and its label (y) are known.
    - Typically referred to as “output” or “target”
    - Denoted as:
        - y if a single number.
        - **y** if it is a vector of different label values such as in multi-label classification.
    - Label space: Y
    - Acquiring labeled data:
        - Sometimes require human labour.
            - Ex: Verifying if an image shows a cat.
        - If labeled data is not possible, unsupervised ML methods must be saught.
- Numeric Label - Regression
    - Label space consists of numerical values like the reals, integers, naturals, ect…
- Categorical Labels - Classification
    - Number of categories: K
        - labels could be values of {1, 2, …, K}
    - Ex: Binary Classification
        - Whether someone has a university degree or not.
- Ordinal Labels
    - an in between for numeric and categorical labels.
    - take on values from an numeric, ordered, and finite set.
    - Ex: Representing qualities with multiple labels.
        - Take a satellite image of an area depicting an area:
            - y = 1: contains no trees
            - y = 2: partially covered by trees
            - y = 3: entirely covered by trees.
            - We might say that label value y = 2 is “larger” than label value y = 1 and label value y = 3 is “larger” than label value y = 2.
- Distinction Between Numeric and Categorical Labels
    - Not well defined.
    - Ex: Binary Classification y∈{-1, 1}
        - Becoming a regression problem:
            - using a label y’ which represents the probability of y being 1.
            - prediction for ŷ’: for the numeric label y’, ŷ can be predicted by setting
                - ŷ := 1 if ŷ′ ≥ 0 and ŷ := −1 otherwise
    - Logistic Regression
        - An in between for regression and classification.
        - A binary classification method, uses same model and linear regression but not the typical average loss function.
- ML methods learn (or search for) a “good” predictor h: X→Y
    - x∈X (features of a data point) as an input
    - ŷ∈Y (predicted label) as an ouput.
        - ŷ = h(x)∈Y
    - Good predictors:
        - ŷ ≈ y or |ŷ - y|→0

### 2.1.3 Scatterplot

- In **ℝ,** we can represent data points **z**^(i) = [x^(i), y^(i)]^T in a 2D plane with axes representing the values of feature x and label y.
    - Visuals can provide a potential relationship between feature x and label y.

### 2.1.4 Probabilistic Models for Data

- **Realization (or: observation, observed value) of a random variable:** the value that is actually observed (what actually happened).
- Interpreting data points as the realization of a random variable (RV).
- Ex: i.i.d. assumption
    - Interprets data points x^(1), . . . , x^(m) as realizations of statistically independent RVs with the same probability distribution p(x).
    - This interpretation allows us to use the properties of the probability distribution to characterize overall properties of entire datasets, i.e., large collections of data points.
- The probability distribution p(x) underlying the data points within the i.i.d. assumption is either known (based on domain expertise) or estimated from data.
- Parameters of a probability distribution:
    - Expected Value/Mean: E(x)
    - Variance: E(x^2) - E(x)^2

## 2.2 The Model

- Goal of an ML Method:
    - Learn a hypothesis map h: X →Y such that y ≈ h(x) = ŷ for any data point.
1. quantify the approximation error.
2. specify what it means for the approximation error to hold for “any” data point.
3. use a simple probabilistic model for data.
- Assume we have found a reasonable hypothesis h where h: X → Y, h(x) = ŷ.
    - This hypothesis can predict the label of any data point given its features.
    - For problems using a finite label space Y (e.g, Y = {−1, 1}), the hypothesis map is a also a classifier.
        - Given Y, we can characterize a particular classifier h(**x**) using its different decision regions:
            - 
                
                ![Screenshot 2025-01-02 at 7.43.32 PM.png](ML%20Basics%20Chapter%202%20-%20Three%20Components%20of%20ML%2016fb5821733180dfa025c4e866055537/Screenshot_2025-01-02_at_7.43.32_PM.png)
                
            - where a∈Y is associated with a specific region R_a.
- A label space Y^X is too large to to be searched over by a practically ML method.
    - Practical ML methods search and evaluate only a tiny subset of all possible hypothesis maps.
        - subset is computationally feasible.
    - ML methods typically use a hypothesis space H that is a small subset of Y^X
        - the choice for the hypothesis space involves a trade-off between computational complexity and statistical properties of the resulting ML methods.
- Underfitting: if an ML method uses a hypothesis space that does not contain any hypotheses maps that can accurately predict the label of any data points.
- Overfitting: hypothesis space is too large and almost perfectly predict the labels of data points in a training set which was used to learn the hypothesis.

### 2.2.1 Parametrized Hypothesis Spaces

- Hypothesis space:
    
    ![Screenshot 2025-01-03 at 12.23.28 PM.png](ML%20Basics%20Chapter%202%20-%20Three%20Components%20of%20ML%2016fb5821733180dfa025c4e866055537/Screenshot_2025-01-03_at_12.23.28_PM.png)
    
    - where
        
        ![Screenshot 2025-01-03 at 12.24.04 PM.png](ML%20Basics%20Chapter%202%20-%20Three%20Components%20of%20ML%2016fb5821733180dfa025c4e866055537/Screenshot_2025-01-03_at_12.24.04_PM.png)
        
    - For classification problems: yˆ=1 for h(w)(x) ≥ 0 and yˆ=−1 otherwise
- Upgrading a Hypothesis Space via Feature Maps
    - Given hypothesis space H, find H’ ⊇ H with more hypothesis maps.
    - Replace the original features **x** of a data point with new (transformed) features **z** = **Φ(x).**
    - Consists of all concatenations of the feature map φ and some hypothesis h∈H.
    - 
        
        ![Screenshot 2025-01-03 at 12.36.39 PM.png](ML%20Basics%20Chapter%202%20-%20Three%20Components%20of%20ML%2016fb5821733180dfa025c4e866055537/Screenshot_2025-01-03_at_12.36.39_PM.png)
        
        - Allows for different combinations of a feature map Φ and “base” hypothesis space H.
        - Requirement: the range of the feature map Φ must belong to the domain of the maps in H.
            - Range(Φ) **⊆** Domain(H)
    - Requires numerical features.

### 2.2.2 The Size of a Hypothesis Space

- Ex: Black and White Pixels
    - 100x10=1000 black and white pixels characterized by a binary label y∈{0, 1}.
    - Model data points using feature space X = {0, 1}^1000 and label space Y={0, 1}
    - Largest possible hypothesis space: H = Y^X
        - Cardinality:  |H| = 2^2^1000
- Infinite Hypothesis Space
    - Cannot use the number of elements as a measure for its size as the number is not well-defined.
    - Instead use the defective dimension d_eff(H) which is the maximum number D∈ℕ such that for all data points (x^(i), y^(i))∈D with different features, we can always find a hypothesis h that fits the labels, y^(i) = h(x^(i)) for i=1…D.

## 2.3 The Loss

- Loss function: L: X × Y × H→ **ℝ**_+ : ((**x**, y), h) → L((**x**, y), h)
- ML methods try to find (learn) a hypothesis that incurs a minimum loss.
- L ((x, y), h) → 0 ⇒ less discrepancy
- The minimization of a loss function that is either non-convex or non-differentiable tends to be computationally much more difficult.
- The choice of loss function might also be guided by probabilistic models for the data generated in an ML application.
- Goals: computationally cheap , statistically robust, interpretable

### 2.3.1 Loss Functions for Numeric Labels

- Squared error loss:
    
    ![Screenshot 2025-01-03 at 1.29.21 PM.png](ML%20Basics%20Chapter%202%20-%20Three%20Components%20of%20ML%2016fb5821733180dfa025c4e866055537/Screenshot_2025-01-03_at_1.29.21_PM.png)
    
    - Pros:
        - Linear predictor maps h(x) = w^Tx, the function is convex and differentiable of the weight vector w.
            - Allows efficient search for optimal predictor.
        - Useful interpretation in terms of a probabilistic model for the features and labels.
    - Cons:
        - Penalizes large |h(x)| values regardless of whether the classification is correct or not.
    - minimizing the square error loss == maximum likelihood estimation
- Absolute error loss |ŷ - y| .
    - Used in  regression problems.
    - Pros:
        - Guides the learning of a predictor.
        - Robust against a few outliers in the training set.
    - Cons:
        - Increased computational complexity of minimizing (non-differentiable) absolute error loss.
- Ex: Binary Classification
    - hypothesis h(x) ∈ **ℝ** maps features x to real numbers
    - ŷ = 1 for h(x) ≥ 0
    - ŷ = -1 for h(x) < 0
    - The sign of h(x) determines the predicted label.
    - The magnitude |h(x)| represents confidence.

### 2.3.2 Loss Functions for Categorical Labels

![Screenshot 2025-01-03 at 4.10.38 PM.png](ML%20Basics%20Chapter%202%20-%20Three%20Components%20of%20ML%2016fb5821733180dfa025c4e866055537/Screenshot_2025-01-03_at_4.10.38_PM.png)

- Ex: 0/1 Loss
    - 
        
        ![Screenshot 2025-01-03 at 4.02.52 PM.png](ML%20Basics%20Chapter%202%20-%20Three%20Components%20of%20ML%2016fb5821733180dfa025c4e866055537/Screenshot_2025-01-03_at_4.02.52_PM.png)
        
    - Properties:
        - Indicates whether ŷ prediction is correct.
        - Appealing statistical interpretation.
        - **non-convex** and **non-differentiable**, making it computationally difficult to optimize.
- Ex: Hinge Loss
    - 
    
    ![Screenshot 2025-01-03 at 4.04.27 PM.png](ML%20Basics%20Chapter%202%20-%20Three%20Components%20of%20ML%2016fb5821733180dfa025c4e866055537/Screenshot_2025-01-03_at_4.04.27_PM.png)
    
    - Properties:
        - Encourages correct classification ( yh(x) ≥ 1 ) with confidence ( |h(x)|  large).
        - Convex but **non-differentiable** for  yh(x) = 1 , requiring advanced methods like **subgradient descent**.
- Ex: Logistic Loss
    - 
        
        ![Screenshot 2025-01-03 at 4.06.20 PM.png](ML%20Basics%20Chapter%202%20-%20Three%20Components%20of%20ML%2016fb5821733180dfa025c4e866055537/Screenshot_2025-01-03_at_4.06.20_PM.png)
        
    - Properties:
        - Convex and **differentiable**, allowing simple gradient-based methods (e.g., gradient descent) for optimization.
        - Commonly used in **logistic regression**.
        - Decreases smoothly with correct classifications and confidence.
- Ex: Squared Error Loss
    - 
    
    ![Screenshot 2025-01-03 at 4.07.59 PM.png](ML%20Basics%20Chapter%202%20-%20Three%20Components%20of%20ML%2016fb5821733180dfa025c4e866055537/Screenshot_2025-01-03_at_4.07.59_PM.png)
    
    - Properties:
        - Performs poorly for classification because it penalizes large  |h(x)|  values even for correct classifications.
        - Shows it preferring poor classifiers.
            
            ![Screenshot 2025-01-03 at 4.09.09 PM.png](ML%20Basics%20Chapter%202%20-%20Three%20Components%20of%20ML%2016fb5821733180dfa025c4e866055537/Screenshot_2025-01-03_at_4.09.09_PM.png)
            

### 2.3.3 Loss Functions for Ordinal Label Values

- Ex: Areal Images of Rectangular Areas of Size
    - 1km x 1km
    - Each data point (rectangular area) is characterized by the feature vector x obtained by stacking RGB values of each image pixel.
    - Each rectangular area is characterized by a label y ∈ {1, 2, 3} where
        - y = 1 means that the area contains no trees.
        - y = 2 means that the area is partially covered by trees.
        - y = 3 means that the are is entirely covered by trees.
        - y = 3 > y = 2 > y = 1
    - Consider two hypotheses that yield 2 and 3, both predictions are wrong.
        - Since order matters, 2 is more correct but with a typical 0/1 loss function, it does not reflect the preference for 2.
        - Modify loss function:
            - 
                
                ![Screenshot 2025-01-03 at 4.16.25 PM.png](ML%20Basics%20Chapter%202%20-%20Three%20Components%20of%20ML%2016fb5821733180dfa025c4e866055537/Screenshot_2025-01-03_at_4.16.25_PM.png)
                

### 2.3.4 Empirical Risk

- The goal of machine learning is to find a hypothesis  h  from a given hypothesis space  H  that minimizes the loss for arbitrary data points. In simpler terms:
    - We want a function  h  that predicts well on new, unseen data.
    - To formalize this, we need a way to measure how good  h  is.
        - This is where the Bayes risk and empirical risk come in.
- Specifying an “arbitrary data point”.
    - Successful approach: using probabilistic models.
- Interpreting Data Points as Realizations of i.i.d
    - AKA: treating the data points in your dataset as being drawn independently from the same underlying probability distribution.
- Bayes Risk
    - The expected value of the loss over all possible data points
        
        ![Screenshot 2025-01-03 at 4.31.20 PM.png](ML%20Basics%20Chapter%202%20-%20Three%20Components%20of%20ML%2016fb5821733180dfa025c4e866055537/Screenshot_2025-01-03_at_4.31.20_PM.png)
        
    - If we learn a hypothesis h* that incurs minimum Bayes risk,
        - measures the **average loss** when applying  h  to random data points  (x, y)  sampled from  p(x, y) .
        - A smaller Bayes risk indicates a better hypothesis h .
    
    ![Screenshot 2025-01-03 at 4.34.39 PM.png](ML%20Basics%20Chapter%202%20-%20Three%20Components%20of%20ML%2016fb5821733180dfa025c4e866055537/Screenshot_2025-01-03_at_4.34.39_PM.png)
    
- Bayes Estimator: a hypothesis that solves the equation
    - This is the **ideal hypothesis** that performs optimally, assuming we know  p(x, y) .
        
        ![Screenshot 2025-01-03 at 4.38.56 PM.png](ML%20Basics%20Chapter%202%20-%20Three%20Components%20of%20ML%2016fb5821733180dfa025c4e866055537/Screenshot_2025-01-03_at_4.38.56_PM.png)
        
    - Notation \hat{L}(h) is shorthand for \hat{L}(h|D)
    - Cons:
        - We typically don’t know p(x,y)
        - Even if p(x, y) is known, solving the optimization problem for h^* can be computationally challenging.
- Empirical Risk Minimization
    - 
        
        ![Screenshot 2025-01-03 at 4.49.53 PM.png](ML%20Basics%20Chapter%202%20-%20Three%20Components%20of%20ML%2016fb5821733180dfa025c4e866055537/Screenshot_2025-01-03_at_4.49.53_PM.png)
        
    - Typically don’t know p(x, y), so we can approximate the Bayes risk using the emperical risk based on a finite dataset D:
        - D = \{(x^{(1)}, y^{(1)}), \dots, (x^{(m)}, y^{(m)})\}
- Connection Between Baye’s Risk and Emperical Risk
    - If the data points in D are i.i.d. samples from p(x, y) then the law of large numbers implies that \hat{L}(h|D) **≈** E(L((x, y), h)) for large m.
- Practical Implications
    - We minimize empirical risk because we do not know  p(x, y) .
    - A larger dataset ( m ) improves the approximation of Bayes risk by empirical risk.
    - The quality of the hypothesis depends on both the loss function  L  and the representativeness of the dataset  D.

- Confusion Matrix
    - Generalizes the concept of 0/1 loss to situations where the diversity of labels varies significantly (imbalanced data).
        - Utilize multiple loss functions.
    - Data set D, feature vectors x^(i), and labels y^(i) ∈ {1,…,k}.
    - 0/1 Loss Function
        - If the dataset D contains mostly data points with one specific label value the average 0/1 loss might obscure the performance of h for data points having of the rare label values.
        - If the average 0/1 loss is very small the hypothesis might perform badly on minority categories.
    - For each pair of two different labels c, c′ ∈ {1, . . . , k}, c ̸= c′, the loss function is defined as:
        - 
            
            ![Screenshot 2025-01-03 at 8.30.41 PM.png](ML%20Basics%20Chapter%202%20-%20Three%20Components%20of%20ML%2016fb5821733180dfa025c4e866055537/Screenshot_2025-01-03_at_8.30.41_PM.png)
            
        - avg loss:
            
            ![Screenshot 2025-01-03 at 8.32.14 PM.png](ML%20Basics%20Chapter%202%20-%20Three%20Components%20of%20ML%2016fb5821733180dfa025c4e866055537/Screenshot_2025-01-03_at_8.32.14_PM.png)
            
    - rows: different values c of the true label of a data point
    - cols: different values c′ delivered by the hypothesis h(x)
    - (c,c′)-th entry of the confusion matrix is Lb(c→c′)(h|D)

- Precision, Recall and F-Measure
    - metrics used to evaluate the performance of classifiers, particularly in imbalanced datasets.
    - **High Recall, Low Precision**:
        - The model captures most positives but also makes many false positive errors.
        - Ex: Detecting objects in an image where missing an object is unacceptable.
    - **High Precision, Low Recall**:
        - The model is very selective in predicting positives, reducing false positives but missing many actual positives.
        - Ex: Detecting spam emails where false alarms (good emails marked as spam) are undesirable.
    - Recall
        
        ![Screenshot 2025-01-03 at 8.43.01 PM.png](ML%20Basics%20Chapter%202%20-%20Three%20Components%20of%20ML%2016fb5821733180dfa025c4e866055537/Screenshot_2025-01-03_at_8.43.01_PM.png)
        
        - Measures the model’s ability to identify all positive cases.
        - High recall = fewer missed positive cases.
    - Precision
        - 
            
            ![Screenshot 2025-01-03 at 8.43.48 PM.png](ML%20Basics%20Chapter%202%20-%20Three%20Components%20of%20ML%2016fb5821733180dfa025c4e866055537/Screenshot_2025-01-03_at_8.43.48_PM.png)
            
        - Measures how accurate the positive predictions are.
        - High precision = fewer false positives.
    - F-Measure
        - We would like to find a hypothesis with both, large recall and large precision.
            - goals are typically conflicting, a hypothesis with a high recall will have small precision.
            - combine the recall and precision of a hypothesis into a single quantity:
            - 
                
                ![Screenshot 2025-01-03 at 8.46.06 PM.png](ML%20Basics%20Chapter%202%20-%20Three%20Components%20of%20ML%2016fb5821733180dfa025c4e866055537/Screenshot_2025-01-03_at_8.46.06_PM.png)
                
    - F_b-Score
        
        ![Screenshot 2025-01-03 at 8.52.13 PM.png](ML%20Basics%20Chapter%202%20-%20Three%20Components%20of%20ML%2016fb5821733180dfa025c4e866055537/Screenshot_2025-01-03_at_8.52.13_PM.png)
        
        - Thus, the F-Measure is essentially the F_1-Score.

### 2.3.5 Regret

- Experts: predictions obtained from some reference methods.
- Regret: quality of a hypothesis h is measured via the difference between the loss incurred by its predictions h(x) and the loss incurred by the predictions of the experts.
    - Goal: minimize, which means learn a hypothesis with a small regret compared to given set of experts.
    - Concept of regret minimization: avoids the need for a probabilistic model of the data to obtain a baseline.
        - This replaces the Bayes risk with the regret relative to given reference methods (the experts)
- Probabilistic Assumptions:
    - When we don’t have any, we cannot use the Bayes risk of the (optimal) Bayes estimator as a baseline (or benchmark).

### 2.3.6 Rewards as Partial Feedback

- When labels are too difficult or costly to deal with, we cannot use them.
    - Thus, we cannot evaluate the loss function for different choices for the hypothesis.
    - So, we must rely on some indirect feedback or “reward” that indicates the usefulness of a particular prediction.