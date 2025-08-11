# Personal-Loan-Prediction
This project predicts personal loan acceptance using the Kaggle 'Bank Personal Loan Modelling' dataset. Various ML models (SVM, KNN, Naive Bayes, Decision Tree, Logistic Regression) are compared with Neural Networks. Decision Tree achieves the highest accuracy.

### Import Packages

### Read Files
<img width="857" height="283" alt="image" src="https://github.com/user-attachments/assets/11b65112-5e62-4237-8345-7544b71d278f" />

### Checking data structure

### Checking missing value and data types

### Drop unnecessary columns

### Pre-processing

### Drop duplicates

### Separate numeric and categorical columns

### Create a function to check the dataframe for columns, data types, unique values, and null values
<img width="309" height="372" alt="image" src="https://github.com/user-attachments/assets/a761252f-3056-44bf-97bf-f206ca6f4532" />

### Visualize Median price of each category
<img width="850" height="372" alt="image" src="https://github.com/user-attachments/assets/3f477a7f-6ee6-4012-a97d-25776e93e644" />

## Continuous Features

<img width="854" height="470" alt="image" src="https://github.com/user-attachments/assets/c9599e1e-bc10-4098-84b4-8f7411e5fcd6" />

## PCA

# Correlation between continuous data
<img width="686" height="591" alt="image" src="https://github.com/user-attachments/assets/101699ee-ea6a-4c1d-9ffd-ecd892168458" />

## Models without standardized data
### Train Test split
### Visualize the proportion between class 1 and class 0
<img width="858" height="440" alt="image" src="https://github.com/user-attachments/assets/701a5d9a-9a48-4ec5-9b5c-8ca59173dd9c" />

## Neural Network
Note:
The number of hidden neurons should be between the size of the input layer and the size of the output layer.
The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
The number of hidden neurons should be less than twice the size of the input layer.

## Neural Network model with 4 hidden nodes
### Performace Evaluation
<img width="251" height="197" alt="image" src="https://github.com/user-attachments/assets/fba9a980-3344-492a-a9f9-eedf9058c5a0" />
<img width="375" height="243" alt="image" src="https://github.com/user-attachments/assets/309b34f7-50c9-422d-b347-018d507b0cdf" />

## Neural Network model with 7 hidden nodes
### Performace Evaluation
<img width="261" height="187" alt="image" src="https://github.com/user-attachments/assets/2bd20536-4720-4515-a44d-00b79a9fbd67" />
<img width="365" height="227" alt="image" src="https://github.com/user-attachments/assets/d6a004ce-3a7a-4bc0-acc1-cc51b4b519f4" />

## SVM
### Performace Evaluation
<img width="265" height="190" alt="image" src="https://github.com/user-attachments/assets/c6089bc1-95ca-4e38-b66d-e4667ad69618" />
<img width="360" height="223" alt="image" src="https://github.com/user-attachments/assets/25081a8a-ef02-449c-b835-2eade6e887b0" />

## Decision Tree
### Performace Evaluation
<img width="263" height="190" alt="image" src="https://github.com/user-attachments/assets/05464d24-371f-4f88-bbed-081829305013" />
<img width="369" height="224" alt="image" src="https://github.com/user-attachments/assets/a5b9db30-47b4-48a6-a3dd-2b0958530e39" />
<img width="766" height="725" alt="image" src="https://github.com/user-attachments/assets/e2f73b93-a905-4347-8bcb-46151e46dc24" />

## Logistic Regression
<img width="228" height="171" alt="image" src="https://github.com/user-attachments/assets/18b2a1de-3bc2-45b1-8ce8-333c3f0f2c4d" />
<img width="330" height="212" alt="image" src="https://github.com/user-attachments/assets/09b82dc7-3f93-46d6-b8b1-13bf6ff0a399" />
<img width="815" height="314" alt="image" src="https://github.com/user-attachments/assets/c0033abb-5b78-4c28-9e15-a0fbae19f9b9" />

## Naive Bayes
### Performace Evaluation
<img width="226" height="169" alt="image" src="https://github.com/user-attachments/assets/c5498c44-8599-4ec2-8191-265f7d18eee6" />
<img width="321" height="198" alt="image" src="https://github.com/user-attachments/assets/8814d4ef-cc2f-48fd-b033-8bf20b20ae58" />

## KNN
<img width="696" height="433" alt="image" src="https://github.com/user-attachments/assets/e482b120-8cc1-483a-a8ec-8a4f46fac362" />

### Performace Evaluation
<img width="222" height="162" alt="image" src="https://github.com/user-attachments/assets/47d7e08f-44ec-452b-b997-31bd53d2ce7e" />
<img width="340" height="201" alt="image" src="https://github.com/user-attachments/assets/1bbfbcfb-c73a-4501-9c83-6b16ce6a492c" />

## Conclusion and Comparison
<img width="805" height="638" alt="image" src="https://github.com/user-attachments/assets/9bdb52af-9782-4047-8b5a-0f14940a6b51" />

## Accuracy:
#### Decision Tree has the highest accuracy, suggesting it makes the fewest errors overall.
#### Naive Bayes has the lowest accuracy, indicating room for improvement.
## Recall:
#### Decision Tree also excels in recall, meaning it effectively identifies true positive cases.
#### Naive Bayes scores zero in recall, implying it fails to identify positive cases.
## Neural Network Performance:
#### Accuracy: The Neural Network shows a strong accuracy of 0.95, indicating it is quite effective in making correct predictions overall.
#### Recall: It has a lower recall at 0.63, which means it may miss identifying some positive instances.
#### While the Neural Network is generally accurate, improving recall would help in better recognizing true positives, potentially improving its effectiveness for the intended task. Adjustments like data normalization, tweaking hyperparameters, or changing the model architecture could potentially enhance performance.
