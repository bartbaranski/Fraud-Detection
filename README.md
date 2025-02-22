<h2>Description</h2>
<p>
  This project focuses on detecting fraudulent transactions using a combination of data preprocessing, oversampling techniques, multiple machine learning models, and an interactive Shiny application for result visualization. The entire process is documented step by step in an R Markdown file, ensuring clarity and reproducibility.
</p>

<h2>Technologies</h2>
<ul>
  <li>R</li>
  <li>Shiny</li>
  <li>R Markdown</li>
  <li>tidyverse</li>
  <li>ROSE</li>
  <li>caret</li>
  <li>randomForest</li>
  <li>xgboost</li>
  <li>rpart</li>
</ul>

<h2>Technologies Description</h2>
<ul>
  <li><strong>R:</strong> The main programming language used for data analysis and model development.</li>
  <li><strong>Shiny:</strong> An interactive web framework used to build dynamic dashboards for data exploration and model evaluation.</li>
  <li><strong>R Markdown:</strong> Used for documenting the entire workflow in a step-by-step manner, combining code, output, and narrative.</li>
  <li><strong>tidyverse:</strong> A collection of R packages (e.g., dplyr, ggplot2, tidyr) for efficient data manipulation and visualization.</li>
  <li><strong>ROSE:</strong> A package for oversampling to balance imbalanced datasets.</li>
  <li><strong>caret:</strong> Provides a unified interface for model training and evaluation across different algorithms.</li>
  <li><strong>randomForest:</strong> Implements the Random Forest algorithm for robust ensemble-based predictions.</li>
  <li><strong>xgboost:</strong> Implements the XGBoost algorithm, a powerful gradient boosting method.</li>
  <li><strong>rpart:</strong> Implements decision tree algorithms based on greedy methods for splitting data.</li>
</ul>

<h2>Features</h2>
<ul>
  <li><strong>Data Preprocessing:</strong> Loads and cleans transaction data from a CSV file, ensuring correct formatting.</li>
  <li><strong>Oversampling:</strong> Balances the dataset using the ROSE package to address the imbalance between fraudulent and non-fraudulent transactions.</li>
  <li><strong>Train-Test Split:</strong> Splits the balanced data into training and testing sets for reliable model evaluation.</li>
  <li><strong>Multiple Models:</strong> Includes Random Forest, Logistic Regression, XGBoost, and Decision Tree models to provide diverse approaches for fraud detection.</li>
  <li><strong>Interactive Dashboard:</strong> An interactive Shiny application that allows users to explore data distributions and evaluate model performance with confusion matrices and key metrics.</li>
  <li><strong>Step-by-Step Documentation:</strong> The entire project is documented in an R Markdown file, providing a comprehensive, reproducible workflow.</li>
</ul>

<h2>Usage</h2>
<ul>
  <li><strong>Deployment:</strong> The Shiny app is deployed on shinyapps.io, allowing easy sharing via a URL in your portfolio or CV.</li>
  <li><strong>Data Exploration:</strong> Users can interactively select variables to visualize their distributions and see summary statistics of the balanced dataset.</li>
  <li><strong>Model Evaluation:</strong> Users can choose from the pre-trained models (loaded from saved RDS files) to view their confusion matrices and performance metrics.</li>
  <li><strong>Documentation:</strong> Detailed, step-by-step instructions and code are provided in the R Markdown file for transparency and reproducibility.</li>
</ul>

<h2>Future Improvements</h2>
<ul>
  <li>Integrate additional machine learning models and ensemble techniques to potentially improve predictive accuracy.</li>
  <li>Optimize resource usage to handle larger datasets or reduce memory consumption.</li>
  <li>Enhance the user interface and user experience of the Shiny application for better accessibility and visual appeal.</li>
  <li>Implement real-time data streaming to enable dynamic fraud detection in live scenarios.</li>
  <li>Add security features such as user authentication and role-based access control.</li>
  <li>Improve logging and error handling for easier debugging and maintenance.</li>
</ul>
