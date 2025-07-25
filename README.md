## Name: MUSINGUZI Medard
## ID: 26601

## 🚖 Uber Fares Analysis – Power BI Project

### 📘 Course: Introduction to Big Data Analytics (INSY 8413)  
**Instructor**: Dr. Eric Maniraguha  
**Institution**: Adventist University of Central Africa  
**Date**: 25,July,2025  

---

## 📌 Project Overview

This project analyzes the **Uber Fares Dataset** to uncover insights about fare patterns, time-of-day effects, and ride behavior. We used **Python (Pandas)** in VS Code for cleaning and exploration, and **Power BI** to design a professional, interactive dashboard that supports data-driven decisions.

---

## 🧾 Analytical Report

### 🟦 1. Introduction

This project aims to explore ride fares, time patterns, and geographic distributions within Uber’s fare dataset using big data analysis techniques. Our goal is to build a **clean, visual, and interactive dashboard** that tells the story behind fare data.

---

### 🟩 2. Methodology

- **📥 Data Collection**: Dataset downloaded from [Kaggle](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset)
  ```python
  # Step 1: Import pandas
    import pandas as pd
  # Step 2: Load the Uber Fares dataset
  df = pd.read_csv(r'C:\Users\user\Downloads\archive\uber.csv')
  # Step 3: Display the first few rows
  print(" First 5 rows of the dataset:")
  print(df.head())

  # Step 4: Display dataset structure and summary info
  print("\n Dataset Info:")
  print(df.info())

  ```
- **🧹 Data Cleaning**:
  - Missing values removed
  - Datetime fields converted
  - Duplicates dropped
  - Outliers detected using IQR
- **🧠 Feature Engineering**:
  - Extracted `hour`, `day`, `month`, and `weekday` from `pickup_datetime`
  - Created `peak/off-peak` time categories
- **📊 EDA**: Descriptive statistics, fare vs distance/time, data distributions
- **📤 Export**: Cleaned dataset saved as `cleaned_uber.csv`
- **📈 Visualization**: Dashboard built in Power BI

---

### 🟨 3. Analysis

- **Summary Statistics**: Mean, median, mode, standard deviation, quartiles
- **Distributions**: Histograms for fare amounts, time breakdowns
- **Outliers**: Detected in `fare_amount` using the IQR method
- **Relationships**:
  - Fare vs. Distance
  - Fare vs. Time of Day
  - Rides across hours, days, and months

---

### 🟥 4. Results

- Fares are higher during **evening hours and weekends**
- Most rides are short-distance, frequent, and concentrated in urban areas
- Fare amount shows moderate correlation with distance
- **Rush hour** and **late-night** periods show spikes in both fare and volume

---

### 🟪 5. Conclusion

The project successfully highlights Uber ride behavior based on time and fare patterns. Clean, structured analysis allowed for building a **clear and functional dashboard** for insight delivery.

---

### 🟫 6. Recommendations

- 🎯 Target drivers and pricing around evening and weekend peaks
- 💬 Use fare surge caps to protect customers during outliers
- 🌍 Focus marketing campaigns on low-demand zones
- 📊 Enhance forecasting models using engineered time features

---

## 📁 Project Structure

