# Advanced Feature Engineering Techniques for EEG Data

## Abstract

This project explores the application of advanced feature engineering techniques to improve the predictive modeling of Central Neuropathic Pain (CNP) development in Spinal Cord Injury (SCI) patients using EEG data. Employing a combination of Mutual Information, Fisher Score, LASSO Regularization, and Forward Sequential Feature Selection, we aimed to enhance the accuracy and efficiency of Support Vector Machines (SVM) and Logistic Regression (LR) models. Our methodology demonstrated significant improvements in model performance, underscoring the value of strategic feature selection in high-dimensional data scenarios. This work contributes to the broader field of medical diagnostics by showcasing the potential of feature engineering to refine predictive analytics in neurological conditions.


## Table of Contents

- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Results](#results)
- [Discussion](#discussion)
- [Conclusions](#conclusions)
- [Installation and Usage](#installation-and-usage)
- [Contributors](#contributors)
- [Acknowledgments](#acknowledgments)
- [License](#License)
- [Contact Information](#contact-information)


## Introduction
This project delves into the application of feature engineering techniques to predict Central Neuropathic Pain (CNP) in individuals with Spinal Cord Injury (SCI) using EEG data. By employing various feature selection methods, we aim to enhance the predictive power of our machine learning models amidst the challenges posed by high-dimensional datasets.

Leveraging a dataset recorded from a 48-electrode Electroencephalogram (EEG), our study showcases the crucial role of feature engineering in improving model accuracy and performance. Through this work, we highlight the transformative potential of carefully selected features in the domain of predictive modeling, especially in medical diagnostics and treatment planning.


## Dataset Description

### Overview

Our study utilizes EEG (Electroencephalogram) data to predict the development of Central Neuropathic Pain (CNP) in individuals with Spinal Cord Injury (SCI). The dataset comprises electrical brain activity recorded from 18 participants at a sampling rate of 250 Hz using a 48-electrode EEG setup. Approximately 50% of the subjects developed CNP, making this dataset crucial for identifying potential CNP development in SCI patients.

### Preprocessing

The raw EEG data underwent several preprocessing steps to ensure the cleanliness and usability of the data for feature selection and modeling. These steps include:
- **Signal denoising**
- **Normalization**
- **Temporal segmentation**
- **Frequency band power estimation**

### Dataset Dimensions

The final dataset dimensions are 180x432, with the subjects divided into two groups: those who developed CNP and those who did not, within six months of data collection. This high-dimensional dataset poses unique challenges for classification algorithms, highlighting the importance of effective feature engineering in our study.

![Screenshot 2024-02-16 114315](https://github.com/MSPK99/EEG-Based-CNP-Prediction-Advanced-Feature-Engineering-Techniques/assets/157824384/1cbe55eb-6f5f-41c6-9054-dbb161e9c5c9)

*Figure: A 2D scatter plot showing three random features out of a set of 180, illustrating the data distribution and potential correlations.*

## Methodology

Our methodology encompasses a systematic approach to feature engineering, selection, and modeling to predict CNP in SCI patients. The process was divided into several key phases:


### 1. Dataset Preprocessing

The EEG dataset, comprising data from 18 participants, underwent several preprocessing steps to ensure data quality and usability:

- **Signal Denoising**: Removing noise to improve signal clarity.
- **Normalization**: Scaling the EEG signals to a standard range.
- **Temporal Segmentation**: Partitioning the data into segments for analysis.
- **Frequency Band Power Estimation**: Calculating the power within specific frequency bands to identify relevant features.

![output](https://github.com/MSPK99/EEG-Based-CNP-Prediction-Advanced-Feature-Engineering-Techniques/assets/157824384/71338dbf-6e39-49b8-bc83-28970d7ac8e8)

*Figure: cumulative distribution of feature i0 separate for each class :  histograms, empirical cumulative distribution functions, scatter plot*

### 2. Feature Selection Methods

To address the high dimensionality of the EEG data (432 features), we applied four primary feature selection techniques to reduce the feature space without sacrificing model performance:

- **Mutual Information (MI)**: A filtering technique measuring the dependency between variables, aiming to retain features with high dependency on the target variable.
- **Fisher Score (FS)**: Another filtering method that ranks features based on their discriminative power between classes.
- **LASSO Regularization (L1)**: An embedded method that adds a penalty to feature coefficients, effectively shrinking less important feature coefficients to zero.
- **Forward Sequential Feature Selection (FSFS)**: A wrapper method that iteratively adds features based on their contribution to model performance, evaluated through F-statistics.

#### Example

**Feature Importance Analysis**

The following graph depicts the Fisher Score calculated for each feature in our dataset. The Fisher Score is a measure of feature importance used in the context of classification problems.

![Fischer score vs feature](https://github.com/MSPK99/EEG-Based-CNP-Prediction-Advanced-Feature-Engineering-Techniques/assets/157824384/13956ab9-49e8-4b83-88f5-300826cddae6)

##### Fisher Score Analysis

- **Higher scores indicate more important features**: As seen in the graph, features on the right have higher Fisher scores, suggesting they are more discriminative for the classification task.
- **Feature Selection**: This analysis is critical for feature selection, as it helps in identifying which features contribute the most to the separation between classes.

### 3. Classification Methods

We employed two primary classifiers to evaluate the predictive power of the selected features:

- **Support Vector Machine (SVM)**: Used for its effectiveness in high-dimensional spaces and its ability to create a clear margin of separation between classes.
- **Logistic Regression**: Chosen for its simplicity, efficiency, and performance in binary classification tasks, especially when the data is linearly separable.

### 4. Model Evaluation and Selection

- **Base Model Performance**: We first established baseline accuracy scores using the full feature set.
- **Reduced Feature Set Performance**: After applying each feature selection method, we re-evaluated the models to compare performance.
- **Hyperparameter Tuning**: Utilizing Grid Search cross-validation, we fine-tuned the models for optimal performance.
- **Evaluation Metrics**: Accuracy, sensitivity, and specificity were key metrics in assessing each model's ability to predict CNP development.

The following flowchart illustrates the comprehensive methodology applied in our analysis. It starts with the initial dataset and branches into two parallel paths for baseline model accuracy assessment using Support Vector Machine (SVM) and Logistic Regression (LR). Subsequently, it demonstrates the application of four feature selection techniques—Mutual Information (MI), Fisher Score (FS), L1 Regularization (L1), and Forward Sequential Feature Selection (FSF)—followed by the re-evaluation of both models. The results are compared based on the accuracy scores derived from the various combinations of feature selection methods and models.


![Screenshot 2024-02-16 111314](https://github.com/MSPK99/EEG-Based-CNP-Prediction-Advanced-Feature-Engineering-Techniques/assets/157824384/c25a79f4-048b-4f99-8a1a-52f2a8a3b71e)


## Results

Our project aimed to refine the feature set for predicting Central Neuropathic Pain (CNP) in Spinal Cord Injury (SCI) patients using EEG data. The methodology involved testing various feature selection techniques and evaluating their effectiveness through classification models, primarily Support Vector Machine (SVM) and Logistic Regression (LR). Here are the crucial findings:

### Baseline Performance

- **Baseline Accuracy**: Before feature selection, SVM and LR models achieved average accuracy scores of 85.56% and 86.11%, respectively.
- **Baseline Specificity and Sensitivity**: The specificity was noted at 85.00% for SVM and 82.50% for LR, with a sensitivity mean of 86.25% and 89.00%, indicating a solid foundation for comparison.


![Screenshot 2024-02-16 113632](https://github.com/MSPK99/EEG-Based-CNP-Prediction-Advanced-Feature-Engineering-Techniques/assets/157824384/fac260c2-e9fb-4677-8fea-cdce7468cf3a)

*Figure: Baseline Confusion Matixes for LR and SVM*

### Performance Post Feature Selection

The application of feature selection methods yielded significant improvements:

- **Mutual Information and Fisher’s Method**: Both methods outperformed others, with mutual information leading to a slight edge in accuracy improvement.

![Mutual Information vs features](https://github.com/MSPK99/EEG-Based-CNP-Prediction-Advanced-Feature-Engineering-Techniques/assets/157824384/f5f946d0-bcf2-40a3-8c1c-2b0000b373f0)


*Figure: A plot showing the impact of mutual information on feature selection.*

- **Forward Selection**: Showed a marked improvement in sensitivity and specificity, suggesting its effectiveness in balancing the model's ability to identify both classes accurately.
- **L1 Regularisation**: Demonstrated substantial benefits in reducing feature dimensionality while maintaining performance metrics within competitive ranges.

![model performance graphs](https://github.com/MSPK99/EEG-Based-CNP-Prediction-Advanced-Feature-Engineering-Techniques/assets/157824384/e252cd87-49db-49d9-abec-8626989e2fc7)

*Figure: Feature Selection Impact on Model Performance*

### Comparative Analysis

- **Accuracy Enhancement**: Post feature selection, the accuracy scores for models using Mutual Information and Fisher's Method saw an uplift to around 88.33% for SVM and varied improvements for LR across different methods, indicating the efficacy of these feature selection techniques in enhancing model performance.
- **Sensitivity and Specificity Gains**: Forward selection stood out in optimizing the trade-off between sensitivity and specificity, thus highlighting its suitability for datasets where both false positives and false negatives carry significant implications.

### Visual Insights

- **Confusion Matrices**: Figures illustrating the baseline and post-feature selection confusion matrices provide a visual confirmation of the improvements in model precision and recall, underscoring the practical benefits of the applied feature selection strategies. Please review the notebooks for illustrative purposes.


## Discussion

Our evaluation of feature selection techniques to predict CNP in SCI patients highlights the critical balance between model accuracy and computational efficiency. Forward Sequential Feature Selection offered notable performance improvements, despite its computational intensity. Lasso Regularisation and Mutual Information gain provided valuable insights into feature relevance, despite their varied impact on model performance.

The study underscores the importance of selecting appropriate feature selection techniques to enhance model accuracy, particularly in high-dimensional medical datasets. Our findings advocate for a methodological diversity and careful consideration of trade-offs in feature selection, contributing valuable insights to the field of medical diagnostics and predictive modeling.

![Screenshot 2024-02-16 114111](https://github.com/MSPK99/EEG-Based-CNP-Prediction-Advanced-Feature-Engineering-Techniques/assets/157824384/b4554ea9-45ec-4cb7-bd24-2edb915833d1)

*Figure: Scatter Plot Distribution of Features before feature engineering was performed*

### Key Points

- **Balance Between Accuracy and Efficiency**: The effectiveness of Forward Sequential Feature Selection in improving model performance underscores the need to balance computational demands with accuracy.
- **Insights from Various Techniques**: Techniques like Lasso Regularisation and Mutual Information offer insights into the relevance of features, highlighting the importance of a diverse methodological approach in feature selection.
- **Importance of Methodological Diversity**: Our results emphasize the value of exploring various feature selection methods to identify the most effective approach for specific datasets, especially those with high dimensionality.
- **Trade-offs in Feature Selection**: The study illustrates the crucial trade-offs between model complexity, computational efficiency, and accuracy, encouraging a more nuanced approach to predictive modeling in medical diagnostics.

### Implications for Future Research

This discussion points to the necessity for ongoing research into efficient and accurate feature selection methods. Future work should aim to refine these techniques, possibly through the development of new algorithms that balance performance with computational demands more effectively. Additionally, further studies could explore the application of these findings in other medical conditions, broadening the impact of this research in the field of healthcare and predictive analytics.

## Conclusions

This project embarked on a journey to enhance the predictability of Central Neuropathic Pain (CNP) development in patients with Spinal Cord Injury (SCI) through the lens of feature engineering. Utilizing EEG data, we explored various feature selection techniques to optimize the performance of machine learning models, namely Support Vector Machines (SVM) and Logistic Regression (LR).

![Screenshot 2024-02-16 113803](https://github.com/MSPK99/EEG-Based-CNP-Prediction-Advanced-Feature-Engineering-Techniques/assets/157824384/c06b6eed-05e0-4c0a-aeeb-d88b7f07ce4f)

*Table: Performance Metrics Post Feature Selection*

### Key Takeaways

- **Feature Selection Efficacy**: Among the techniques evaluated, Mutual Information, Fisher Score, LASSO Regularization, and Forward Sequential Feature Selection each demonstrated unique strengths in refining the feature set, thereby improving model accuracy and interpretability.
- **Model Performance**: The project confirmed that judicious feature selection significantly impacts model performance. Notably, models enhanced through Forward Sequential Feature Selection showed a balanced improvement in both sensitivity and specificity, underscoring the importance of selecting relevant features for predictive modeling in high-dimensional datasets.
- **Computational Considerations**: While some methods like Forward Sequential Feature Selection offered substantial performance gains, they also highlighted the computational demands associated with exhaustive feature searches. This trade-off between computational efficiency and model accuracy is a critical consideration for future applications.

### Implications for Future Work

The findings underscore the potential of feature engineering in transforming raw, high-dimensional datasets into more manageable, insightful forms that can significantly enhance model performance. This project lays the groundwork for further research into feature selection techniques, especially in the context of medical data, where accurate predictions can substantially influence patient outcomes.

### Closing Thoughts

Our exploration into feature engineering for predicting CNP in SCI patients not only demonstrated the tangible benefits of various feature selection methods but also highlighted the intricate balance between model complexity and performance. As we move forward, the insights garnered from this project will inform more nuanced approaches to predictive modeling, emphasizing the critical role of feature selection in the era of big data and machine learning.


## Installation and Usage

This project is implemented in Python and requires the following libraries: NumPy, Scikit-learn, Matplotlib, CSV, Seaborn, and DateTime. Below are the instructions to set up your environment and run the project.

### Prerequisites

Ensure you have Python 3.6+ installed on your system. You can download Python from the [official website](https://www.python.org/downloads/).

### Setting Up Your Environment

#### Create a Virtual Environment (Optional but Recommended):

1. Navigate to your project's directory in the terminal.
2. Run (replace `myenv` with your preferred environment name).
   ```bash
   python3 -m venv myenv
   ``` 
3. Activate the virtual environment:
   - On Windows: `myenv\Scripts\activate`
   - On macOS/Linux: `source myenv/bin/activate`

#### Install Required Libraries:

1. Ensure your virtual environment is activated.
2. Install the required libraries by running 
   ```bash
   pip install numpy scikit-learn matplotlib seaborn
   ```

### Running the Code

#### Clone the Repository:

1. Clone this project repository to your local machine using `git clone https://github.com/MSPK99/EEG-Based-CNP-Prediction-Advanced-Feature-Engineering-Techniques.git`.
2. Note: Ensure you're extracting the feature set from "aiml_final_dependencies-main". You don't have to worry about this step as we already have 3 csv files with details that entails the .h5 information processable for the task.

#### Navigate to the Project Directory:

1. Change your current directory to the cloned repository's directory in the terminal.

#### Execute the Notebooks:

- If you're using Jupyter Notebooks or JupyterLab:
  1. Start Jupyter by running `jupyter notebook` or `jupyter lab` in your terminal.
  2. Open the .ipynb files and run the cells in sequence.
- If you're using another IDE, ensure it supports .ipynb files and execute the code accordingly.

### Usage Notes

- The code files are structured to guide you through the process of loading the dataset, applying feature selection methods, and evaluating model performance.
- Also, have a look at the "cv performance metrics" folder to better understand the results.



## Contributors

The following individuals have contributed to this project with their respective roles:

- **ID: 2815755M**: Took the lead in coding, resolving bugs, and troubleshooting issues related to the codebase, ensuring the smooth execution of the project's computational tasks.
- **ID: 2805927K**: Responsible for writing the report and visualizing output graphs, ensuring that the findings were clearly communicated and supported by appropriate visual data representations.
- **ID: 2395481A**: Played a crucial role in gathering information regarding various models, providing the team with valuable insights and assisting in the evaluation of the best-performing models.
- **ID: 2740593A**: Focused on identifying the best hyperparameters for the models, contributing to the fine-tuning process which is critical for the success of machine learning projects.
- **ID: 2826798W**: Helped in compiling and gathering outputs from the models, assessed the best models for the project requirements, ensuring the selection of the most suitable algorithms.

## Acknowledgments

We extend our heartfelt thanks to all who have played a part in this project's journey. Special appreciation goes to the University of Glasgow's academic and technical staff for their invaluable expertise. Acknowledgment is also due to the creators and maintainers of the machine learning tools used in our analysis, including PathologyGAN, ResNet50, InceptionV3, and VGG16. Lastly, we're grateful to our professors for their constructive criticism and feedback, which significantly shaped our case study approach.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact

For inquiries or proposals, contact me at praneethkumar.m@yahoo.com.
