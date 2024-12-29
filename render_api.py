"""
This script is used to test the API endpoints of the income prediction model.
"""

import requests
import json



def render_api(url):
    response = requests.get(url)
    print(response.json())
    print(response.status_code)
    
def render_api_predict(url, data):
    """Test prediction endpoint"""
    try:
        response = requests.post(
            url + "predict", 
            json=data
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":

    url = "https://udacity-project-deploying-a-scalable-ml.onrender.com/"

    sample_data = {
        "age": 42,
        "workclass": "Private",
        "fnlgt": 159449,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 5178,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    render_api(url)
    render_api_predict(url, sample_data)

