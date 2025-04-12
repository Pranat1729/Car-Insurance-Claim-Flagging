# 🚗 Car Insurance Claim Flagging

This project builds a machine learning classifier to predict whether a customer is likely to file a car insurance claim. Using customer demographic and behavioral data, the model helps insurance companies identify high-risk customers and manage risk more effectively.

## 📂 Dataset

The dataset used is `car_insurance_claim.csv`, which contains 12 columns:

- `Id`: Unique identifier for each customer
- `Age`: Age of the customer
- `Gender`: Male/Female
- `Driving_License`: 1 if the customer has a license, 0 otherwise
- `Previously_Insured`: 1 if previously insured, 0 otherwise
- `Vehicle_Age`: Age group of the vehicle (`< 1 Year`, `1-2 Year`, `> 2 Years`)
- `Vehicle_Damage`: 1 if the vehicle was previously damaged, 0 otherwise
- `Annual_Premium`: Premium paid annually
- `Policy_Sales_Channel`: Categorical code representing the sales channel
- `Vintage`: Days since the customer was associated with the company
- `Region_Code`: Encoded region identifier
- `Response`: Target variable (1 = customer filed a claim, 0 = did not)

## 🧠 Workflow Summary

### 🔹 Preprocessing
- Encoded categorical columns (`Gender`, `Vehicle_Age`, `Vehicle_Damage`)
- Dropped the `Id` column as it's not informative
- Feature matrix `X` and label `y` created from relevant columns
- Dataset split into training and testing sets using `train_test_split`

### 🔹 Model Training
Implemented and trained:

- **Decision Tree Classifier**
- **Random Forest Classifier**

Both models were trained on the same preprocessed dataset and evaluated on test data.

### 🔹 Evaluation Metrics
- Accuracy Score
- Precision, Recall, F1-Score
- Confusion Matrix

## 📈 Results

| Model               | Accuracy |
|--------------------|----------|
| XGBClassifier      | ~99.8    |

XGBClassifer consistently showed better generalization and fewer false positives.

## 🚀 Future Enhancements

- Perform hyperparameter tuning with GridSearchCV
- Add ROC curves and AUC for better classification evaluation
- Incorporate more robust preprocessing (e.g., StandardScaler for continuous features)
- Explore advanced models like XGBoost or CatBoost
- Build a Streamlit app for user-friendly prediction

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-idea`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-idea`)
5. Open a Pull Request

Let’s build something valuable together.

## 📬 Contact

- **Email**: pranat32@gmail.com  
- **GitHub**: [Pranat1729](https://github.com/Pranat1729)  
- **LinkedIn**: [linkedin.com/in/Pranat1729](https://www.linkedin.com/in/pranat-sharma-a55a77168/)

---


