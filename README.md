# 🌍 **Climate Hero**

Welcome to **Climate Hero**, your go-to tool for visualizing energy consumption and CO2 emissions trends. Designed to empower users with insights, **Climate Hero** predicts future energy usage and emissions trends, enabling informed decisions toward a sustainable future.

---

## 📖 **Table of Contents**
- [Introduction](#introduction)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [How It Works](#how-it-works)
- [Screenshots](#screenshots)
- [Setup](#setup)
- [How It Helps](#how-it-helps)
- [Contributing](#contributing)

---

## ✨ **Introduction**

**Climate Hero** is an interactive web app that leverages machine learning to provide accurate predictions of energy consumption and CO2 emissions for various countries. With visualizations that make complex data easy to understand, this tool fosters awareness about global energy trends and helps users explore solutions for a greener planet.

🔮 **Key Features**:
- **Predictions up to 2030** for energy consumption and CO2 emissions.
- **Normalized graphs** comparing trends side-by-side.
- **Historical data visualizations** with tables and scatter plots.

---

## 🔑 **Features**

- 🌍 **Dynamic Filtering**: Choose a country to analyze its energy and emission trends.
- 📊 **Interactive Visualizations**:
  - Normalized graphs for energy consumption and CO2 emissions.
  - Individual prediction graphs for energy consumption and CO2 emissions.
- 🧠 **Machine Learning Models**:
  - **Prophet** for energy consumption predictions.
  - **Random Forest** for CO2 emissions forecasting.
- 📈 **Historical Data**: Explore past trends with scatter plots and data tables.

---

## 🛠️ **Tech Stack**

| **Technology**  | **Purpose**                                                   |
|------------------|---------------------------------------------------------------|
| **Python**       | Data processing and application logic.                        |
| **Pandas**       | Data manipulation and cleaning.                               |
| **Matplotlib**   | Creating interactive visualizations.                          |
| **Prophet**      | Forecasting future energy consumption trends.                 |
| **Scikit-learn** | Machine learning model to predict CO2 emissions.              |
| **Streamlit**    | Building an interactive and user-friendly web application.    |
| **NumPy**        | Data computations and array handling.                         |

---

## 🔄 **How It Works**

**Input**:
- Upload or access preloaded energy consumption and CO2 emission datasets.

**Process**:
1. **Load Data**: Process the dataset for analysis.
2. **Predictions**:
   - Use **Prophet** for energy consumption predictions.
   - Use **Random Forest** for CO2 emission predictions.
3. **Visualize**:
   - Scatter plots and tables for historical data.
   - Line graphs for future predictions.

**Output**:
- **Historical Data Table**: Overview of past energy usage.
- **Scatter Plot**: Visualization of historical energy consumption by type.
- **Prediction Graphs**:
  - Normalized graph comparing energy and CO2 emissions.
  - Individual graphs for energy consumption and CO2 predictions.

---

## 🖼️ **Screenshots**

### 🌍 Homepage
_**![image](https://github.com/user-attachments/assets/dfdb411f-a7bd-46ab-96f0-0ab13cf575ca)
**_

### 📊 Historical Data
_![image](https://github.com/user-attachments/assets/f060a252-4020-4ca5-ac26-3a951b3b8a21)
_

### 🔮 Energy Consumption by type Graph
_![image](https://github.com/user-attachments/assets/7a9f3a65-c97e-40fd-adfa-7ffff3669e39)
_

### 📈 Energy and CO2 Predictions
_![image](https://github.com/user-attachments/assets/1b6e681b-329d-4705-a0cf-718091dcca34)
_

---

## ⚙️ **Setup**

Follow these steps to set up and run **Climate Hero**:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/KevinShilla/ClimateHero.git
   cd Climate-Hero
   ```

   **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   **Run the App:**
   ```bash
   streamlit run app.py
   ```

**Access the App**:
Open the local URL provided in the terminal (e.g., http://localhost:8501) to explore the app.

**OR**
Go to: https://climatehero.streamlit.app/

💡 How It Helps
Climate Hero is designed to serve a wide audience:

Policy Makers: Use predictions to implement data-driven sustainability initiatives.

Researchers: Analyze trends in energy consumption and CO2 emissions for environmental studies.

Individuals: Gain insights into energy and CO2 trends to make informed lifestyle choices.
