Data Source: WSDM - KKBox's Churn Prediction Challenge
https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data


The raw dataset contains four tables. We merged them into a single table and uploaded it to the Google drive, via link: https://drive.google.com/file/d/1WDfh8HLYOtUNuhRZqKCScd1qb4l9sqyj/view?usp=sharing


Code directory: https://github.com/yingliu02/DATA240_Project/tree/main


Instruction on how to run your code:

Merge_raw_data.py: this file is used to merge the four tables in the raw dataset based on user id. It was edited using Visual Studio Code, and can be run by command line `python3 merge_raw_data.py`.

Data_visualization1.py and data_visualization2.py: these two files are used for data analysis and visualization on the raw dataset. They are edited using VS Code, and can be run by command line `python3 data_visualization1.py` and `python3 data_visualization2.py`.

Classic_ml_models.py: this file is used to train and evaluate the classic machine learning models, such as k-Nearest Neighbors, Decision Tree, Random Forest, AdaBoost, and XGBoost, on imbalanced training data and balanced training data. It was edited using VS Code, and can be run by command line `python3 classic_ml_models.py`.

Neural Network models: run `python3 models/final/run.py` and follow the input prompts to run either the simple or enhanced neural network on the balanced or unbalanced data


Contribution:

Kody Low: Literature review; data collection; data analysis and visualization; data preprocessing and preparation; developing and evaluating classic machine learning models and Neural Network models; project report and presentation slides.

Shreya Singh: Literature review; data collection; data analysis and visualization; data preprocessing and preparation; developing and evaluating classic machine learning models and Neural Network models; project report and presentation slides.

Ying Liu:Literature review; data collection; data analysis and visualization; data preprocessing and preparation; developing and evaluating classic machine learning models and Neural Network models; project report and presentation slides.
