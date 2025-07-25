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
### Output
![<img width="619" height="578" alt="out1" src="https://github.com/user-attachments/assets/4b60eaee-72a9-42fc-87d9-aad52a4dea1d" />
]
- **🧹 Data Cleaning**:
  - Missing values removed
  - Datetime fields converted
  - Duplicates dropped
    ```python
    import pandas as pd

     # Load the dataset
     df = pd.read_csv('archive/uber.csv')

     # Convert pickup_datetime to datetime if applicable
     if 'pickup_datetime' in df.columns:
     df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')

     # 1. Show missing values before cleaning
     print("Missing values before cleaning:")
     print(df.isnull().sum())

     # 2. Drop rows with any missing values
     df_cleaned = df.dropna()

     # 3. Show missing values after cleaning
     print("\nMissing values after dropping rows with nulls:")
     print(df_cleaned.isnull().sum())

     # 4. Remove duplicate rows
     duplicates = df_cleaned.duplicated().sum()
     print(f"\nDuplicate rows found: {duplicates}")
     df_cleaned = df_cleaned.drop_duplicates()

     # 5. Reset the index
     df_cleaned.reset_index(drop=True, inplace=True)

     # 6. Display cleaned dataset shape
     print("\nCleaned dataset shape:", df_cleaned.shape)

      # 7. Save cleaned dataset for Power BI
      df_cleaned.to_csv('cleaned_uber.csv', index=False)
      print("\nCleaned dataset saved as 'cleaned_uber.csv'")

    ```
  ### Output
  ![<img width="487" height="558" alt="out2" src="https://github.com/user-attachments/assets/fa069864-5859-4350-974d-b7d4cb961596" />
]
  - Outliers detected using IQR,Mean, median, mode, standard deviation, quartiles
    ```python
    import pandas as pd

    # Load the cleaned dataset
    df = pd.read_csv('cleaned_uber.csv')

    # 1. Descriptive statistics for numerical columns
    print("Descriptive statistics:")
    print(df.describe())

     # 2. Mean, median, mode, std
     print("\nMean values:")
     print(df.mean(numeric_only=True))

     print("\nMedian values:")
     print(df.median(numeric_only=True))

     print("\nMode values:")
     print(df.mode(numeric_only=True).iloc[0])  # take the first mode row

     print("\nStandard deviation:")
     print(df.std(numeric_only=True))

     # 3. Quartiles (Q1, Q2, Q3) and IQR
     Q1 = df.quantile(0.25, numeric_only=True)
     Q2 = df.quantile(0.50, numeric_only=True)
     Q3 = df.quantile(0.75, numeric_only=True)
     IQR = Q3 - Q1

      print("\nQuartiles:")
      print("Q1 (25%):\n", Q1)
      print("Q2 (50% / Median):\n", Q2)
      print("Q3 (75%):\n", Q3)
      print("IQR (Q3 - Q1):\n", IQR)

      # 4. Data range (max - min)
      data_range = df.max(numeric_only=True) - df.min(numeric_only=True)
      print("\nData Range:")
      print(data_range)

       # 5. Outlier detection using IQR method
       print("\nOutlier Summary (using IQR):")
       for col in df.select_dtypes(include='number').columns:
       lower_bound = Q1[col] - 1.5 * IQR[col]
       upper_bound = Q3[col] + 1.5 * IQR[col]
       outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
       print(f"{col}: {len(outliers)} outlier(s)")

    ```
### Output
![<img width="678" height="579" alt="out3" src="https://github.com/user-attachments/assets/7fad57d0-1bb1-4997-b2c2-005f450d7e2d" />
]
- **🧠 Feature Engineering**:
  - Extracted `hour`, `day`, `month`, and `weekday` from `pickup_datetime`
  - Created `peak/off-peak` time categories
- **📊 EDA**: Descriptive statistics, fare vs distance/time, data distributions
  ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns

   # Load dataset
   uber = pd.read_csv('archive/uber.csv')

   # Clean column names from trailing/leading spaces
   uber.columns = uber.columns.str.strip()

    # Convert pickup_datetime to datetime and extract hour
    uber['pickup_datetime'] = pd.to_datetime(uber['pickup_datetime'], errors='coerce')
    uber['hour'] = uber['pickup_datetime'].dt.hour

    # Define haversine function to calculate distance between lat/lon pairs
    def haversine(lat1, lon1, lat2, lon2):
      lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
       dlat = lat2 - lat1
       dlon = lon2 - lon1
       a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
       c = 2 * np.arcsin(np.sqrt(a))
       r = 6371  # Earth radius in km
       return c * r

    # Calculate distance if 'distance' column not present
    if 'distance' not in uber.columns:
      required_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
       if all(col in uber.columns for col in required_cols):
          uber['distance'] = haversine(
              uber['pickup_latitude'],
              uber['pickup_longitude'],
              uber['dropoff_latitude'],
              uber['dropoff_longitude']
          )
      else:
          print("Warning: Distance cannot be calculated — missing latitude/longitude columns.")

      # ---- b. Fare distribution visualizations ----
      plt.figure(figsize=(18, 5))

      plt.subplot(1, 3, 1)
      sns.histplot(uber['fare_amount'], bins=50, kde=False, color='skyblue')
      plt.title('Fare Amount Distribution')
      plt.xlabel('Fare Amount')
      plt.ylabel('Count')

       plt.subplot(1, 3, 2)
       sns.boxplot(x=uber['fare_amount'], color='lightgreen')
       plt.title('Fare Amount Boxplot')

       plt.subplot(1, 3, 3)
       sns.kdeplot(uber['fare_amount'], shade=True, color='salmon')
       plt.title('Fare Amount Density Plot')
       plt.xlabel('Fare Amount')

       plt.tight_layout()
       plt.show()

       # ---- c. Relationships between variables ----

       # 1. Fare vs Distance scatter plot (if distance column exists)
        if 'distance' in uber.columns:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x='distance', y='fare_amount', data=uber, alpha=0.3)
            plt.title('Fare Amount vs Distance Traveled')
            plt.xlabel('Distance (km)')
            plt.ylabel('Fare Amount')
            plt.show()
       else:
             print("Skipping Fare vs Distance plot - no distance data available.")

          # 2. Fare vs Time of Day (hour)
           plt.figure(figsize=(12, 6))
           sns.boxplot(x='hour', y='fare_amount', data=uber)
           plt.title('Fare Amount vs Time of Day')
           plt.xlabel('Hour of Day')
           plt.ylabel('Fare Amount')
           plt.show()

            # 3. Correlation matrix heatmap for key variables
            cols_for_corr = ['fare_amount']
            if 'distance' in uber.columns:
                cols_for_corr.append('distance')
            if 'hour' in uber.columns:
                cols_for_corr.append('hour')

            corr = uber[cols_for_corr].corr()

            plt.figure(figsize=(6, 4))
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation Matrix')
            plt.show()

  ```
### Screenshots of output diagrams
![<img width="806" height="247" alt="code Dia1" src="https://github.com/user-attachments/assets/8a7af29f-b651-4cc2-a2ab-11ec9ebef2c6" />]
![<img width="743" height="588" alt="code Dia2" src="https://github.com/user-attachments/assets/29ad62be-a9a8-4839-8539-200fa803734b" />
]
![<img width="828" height="474" alt="code Dia3" src="https://github.com/user-attachments/assets/af18a378-9fd4-4244-835d-3bfd847bc648" />]
![<img width="575" height="433" alt="code Dia4" src="https://github.com/user-attachments/assets/52262eca-8ce3-47ae-9b3b-25f38fc23e07" />
]


- **📤 Export**: Cleaned dataset saved as `cleaned_uber.csv`
  ```python
     import pandas as pd

     # Assuming you've already cleaned your data as df_cleaned
     # If not, reload the cleaned version from previous steps
     # df_cleaned = pd.read_csv('archive/uber.csv')
     # (Perform cleaning steps again if necessary)

     # Export the cleaned DataFrame to CSV
     df_cleaned.to_csv('cleaned_uber.csv', index=False)

     print("Cleaned dataset successfully exported as 'cleaned_uber.csv'")

  ```
### 🟥 4. Results and  Dashboard built in Power BI

- Fares are higher during **evening hours and weekends**
- Most rides are short-distance, frequent, and concentrated in urban areas
- Fare amount shows moderate correlation with distance
- **Rush hour** and **late-night** periods show spikes in both fare and volume
### Screenshot of Charts created
![<img width="968" height="531" alt="charts" src="https://github.com/user-attachments/assets/ee485c01-3109-43a8-af19-f0dbcd9d8478" />
]
### Professional Dashboards built in Power BI
![<img width="965" height="527" alt="Dashboard1" src="https://github.com/user-attachments/assets/7b0c4a9d-a8c9-4c78-9017-e60ba75d51aa" />
]
![<img width="957" height="519" alt="Dashboard2" src="https://github.com/user-attachments/assets/116c8036-e2ac-492c-b744-69c2d8238a97" />
]

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

## THANK YOU

