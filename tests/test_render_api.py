import requests
import json

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
}

data = json.dumps(sample_data)


def test_render_api():
    response = requests.get(url)
    print(response.json())
    print(response.status_code)
    try :
        assert response.status_code == 200
        assert response.json()["message"] == "Welcome to the income prediction API!"
    except Exception as e:
        print(e)

def test_render_api_predict():
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

    response = requests.post(url + "predict", json=sample_data)
    print(response.json())
    try :
        assert response.status_code == 200
        assert response.json()["prediction"] in [" <=50K", " >50K"]
    except Exception as e:
        print(e)


def test_render_api_predict_sup50k():
    sample_data = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 14084,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    }

    response = requests.post(url + "predict", json=sample_data)
    print(response.json())
    try :
        assert response.status_code == 200
        assert response.json()["prediction"] == " >50K"
    except Exception as e:
        print(e)

def test_render_api_predict_inf50k():
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

    response = requests.post(url + "predict", json=sample_data)
    print(response.json())
    try :
        assert response.status_code == 200
        assert response.json()["prediction"] == " <=50K"
    except Exception as e:
        print(e)