# NoShowAppointments
This is the project code used for a salary prediction project using standard Human Resource Data.  The problem statement being addressed in the project is, "How can we better predict the expected salary for various positions such that the offerred salary can align with the market?"

# Data Used for Analysis
1. HR Salary Information

# Pre-requisites

Option 1: WSL (Windows Sub-Linux)

1. Enable [WSL](https://winaero.com/blog/enable-wsl-windows-10-fall-creators-update/) in windows 
2. Install Ubuntu App from Windows Store
3. Create Login and sudo password for Linux

Option 2: Google-colab

1. Login to [google colab](https://colab.research.google.com/notebooks/welcome.ipynb)
2. Copy forked GitHub files to google colab
3. Run code 

# Getting Started 

1. Open Windows Sub Linux (Ubuntu App)

2. Run the following command

```sh
git clone https://github.com/narquette/salarypredictionportfolio
```

3. Change install script to executable and run install file

```sh
chmod +x prereq_install.sh
./prereq_install.sh
```

4. Open Jupyter Notebook

```sh
jupyter notebook --no-browser
```
5. [Copy URL from command line](https://www.screencast.com/t/JgVmAL6wC)

6. Run Salary Prediction Notebook.ipynb in the Code folder

# Risk Salary Prediction App

No Show Prediction

1) Go to [Heroku App](https://mysalarypred.herokuapp.com/)
2) Enter in the following values:
      Miles from Metropolis = 45
      Years Experience = 10
      Industry = Health
3) View Salary Prediction:
      "The predicted salary is 157420.0"

# Folder Overview

Code 
- Salary Prediction Notebook (all of the code required to produce a final model)
- HelperFile.py (contains the machine learning class needed to run in the Salary Prediction Notebook

Data
- Original (original salary data)
- Cleaned (cleaned salary data)
- Prediction (the predicted salary data for the test data)

Logs
- Previous Model Logs and Where New Logs Information will be places

Models
- The final model produced from running the notebook

Visualizations 
- Visualizations produced in the EDA (exploratory data analysis phase)
- Pandas Profile HTML file for the original data set
