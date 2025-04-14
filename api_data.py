# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 19:25:12 2025

@author: me
"""

import requests
import pandas as pd

# API URL from crosslibs/covid19-india-api
API_URL = "https://api.covid19india.org/v4/min/data.min.json"

# Fetch the data from the API
response = requests.get(API_URL)

if response.status_code == 200:
    data = response.json()
    
    # Extract relevant state-wise data
    records = []
    for state, details in data.items():
        if "total" in details:
            records.append({
                "State": state,
                "Confirmed": details["total"].get("confirmed", 0),
                "Recovered": details["total"].get("recovered", 0),
                "Deaths": details["total"].get("deceased", 0),
                "Vaccinated": details["total"].get("vaccinated", 0)
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Save to CSV
    csv_filename = "covid19_india_statewise.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Data saved successfully as {csv_filename}")

else:
    print("Failed to fetch data. Status Code:", response.status_code)
