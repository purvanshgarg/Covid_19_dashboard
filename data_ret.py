# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:39:24 2025

@author: me
"""
import pandas as pd
data = pd.read_csv("India_COVID_Second_Wave.csv")
data = data[data.State != 'Unknown']



# ##Get data in desired format
# import pandas as pd
# import os
# from datetime import datetime, timedelta

# # Define the folder containing the downloaded CSVs
# data_folder = "Data"

# # Define the date range for the second wave (March to June 2021)
# start_date = datetime(2021, 3, 1)
# end_date = datetime(2021, 6, 30)
# date_range = pd.date_range(start=start_date, end=end_date)

# # Create an empty DataFrame to store the cleaned data
# columns = ["Date", "State", "Confirmed", "Deaths", "Recovered"]
# final_df = pd.DataFrame(columns=columns)

# # Loop through the date range and process each file
# for date in date_range:
#     file_name = date.strftime("%m-%d-%Y") + ".csv"
#     file_path = os.path.join(data_folder, file_name)

#     if os.path.exists(file_path):
#         df = pd.read_csv(file_path)
        
#         # Filter for India data
#         india_df = df[df["Country_Region"] == "India"]
#         india_df = india_df[["Last_Update", "Province_State", "Confirmed", "Deaths", "Recovered"]]

#         # Rename columns
#         india_df.columns = ["Date", "State", "Confirmed", "Deaths", "Recovered"]
#         india_df["Date"] = pd.to_datetime(india_df["Date"]).dt.date

#         # Append to final DataFrame
#         final_df = pd.concat([final_df, india_df], ignore_index=True)

# # Save the cleaned dataset
# final_df.to_csv("India_COVID_Second_Wave.csv", index=False)
# print("Data processing complete. File saved as 'India_COVID_Second_Wave.csv'")

# ##Download wave 2 data 
# import os
# import requests
# from datetime import datetime, timedelta

# # Folder to save CSV files
# save_folder = "Data"
# os.makedirs(save_folder, exist_ok=True)

# # Define start and end dates
# start_date = datetime(2021, 3, 1)
# end_date = datetime(2021, 6, 30)

# # Loop through each date
# current_date = start_date
# while current_date <= end_date:
#     # Format date to match GitHub file naming convention (MM-DD-YYYY)
#     file_name = current_date.strftime("%m-%d-%Y") + ".csv"
#     file_url = f"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/{file_name}"
    
#     # Define local file path
#     file_path = os.path.join(save_folder, file_name)
    
#     try:
#         # Download the file
#         response = requests.get(file_url)
#         if response.status_code == 200:
#             with open(file_path, "wb") as file:
#                 file.write(response.content)
#             print(f"Downloaded: {file_name}")
#         else:
#             print(f"File not found: {file_name}")
    
#     except Exception as e:
#         print(f"Error downloading {file_name}: {e}")

#     # Move to the next day
#     current_date += timedelta(days=1)

# print("Download complete!")


# import requests

# # Rootnet API endpoint for state-wise data
# url = "https://api.rootnet.in/covid19-in/stats/history"

# # Fetch data
# response = requests.get(url)
# data = response.json()

# # Save data to a file
# import json
# with open("covid19_rootnet_data.json", "w") as file:
#     json.dump(data, file)

# import pandas as pd

# # Load data
# with open("covid19_rootnet_data.json", "r") as file:
#     data = json.load(file)

# # Extract state-wise data
# state_data = data["data"]["regional"]

# # Convert JSON to DataFrame
# df = pd.DataFrame(state_data)

# # Clean and transform data
# df["date"] = pd.to_datetime(df["day"])  # Convert date column
# df["state"] = df["loc"].str.title()  # Standardize state names