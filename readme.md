# Diagnose Parkinson’s Disease

The technique we are going to apporche for diagnosing the Parkinson’s disease using voice samples from the patient is Linear Regression.
We build a library around this, which is used in the Django App for making an API wrapped around the detection library

## Setup and Installation DjangoApp

The first thing to do is to clone the repository:

```sh
$ git clone https://github.com/galoscar07/api_parkinson.git
$ cd api_parkinson
```

Create a virtual environment to install dependencies in and activate it:

```sh
$ virtualenv2 --no-site-packages env
$ source env/bin/activate
```

Then install the dependencies:

```sh
(env)$ pip install -r requirements.txt
```
Note the `(env)` in front of the prompt. This indicates that this terminal
session operates in a virtual environment set up by `virtualenv2`.

Once `pip` has finished downloading the dependencies:
```sh
(env)$ cd api_parkinson
(env)$ python manage.py runserver
```
And navigate to [`http://localhost:8000/results/`]('http://127.0.0.1:8000/results/').

## Algorithm

In order to run only the detection algorithm you can put at the end of the `training_model.py` the following code:
```py
if __name__ == '__main__':
    dataSet = DatasetCreator()
    response = dataSet.predict(wav_path="../path/to/file")
    print(response['success']) # prints true or false depending on the prediction
    print(response['data']) # prints the predicted resonse value
```

## File structure
Down below can be seen the file structure of the API and the algorithm
```py
    - algo
        |-- dataset
            |-- dataset_our_voices
                |...
            |-- dataset_voices
                |-- read_text
                    |...
                |-- spontaneous_dialogue
                    |...
            |-- processed_results.csv
            |-- trained_model.sav
            ("The 2 above files can be missing, but will be generated")
        |-- __init__.py
        |-- training_model.py
    - media ("folder for django to store the saved media")
    - parkinson ("django root app folder")
        |-- __init__.py
        |-- settings.py
        |-- urls.py
        |-- wsgi.py
        |-- asgi.py
    - results ("django app")
        |-- migrations
            |...
        |-- __init__.py
        |-- admin.py
        |-- apps.py
        |-- models.py
        |-- urls.py
        |-- serializers.py
        |-- views.py
        |-- test.py
    - API_Parkinson.postman_collection.json
    - db.sqlite3
    - manage.py
    - requirements.txt
```

## Library Explained
### Algo
```training_model.py```
As the steps involved into approching this, we will use a bank of audio files of healthy and Parkinson's suffereres (which can be found [here]('https://zenodo.org/record/2867216#.Ym6wWPNBxpT')). We will process each audio file, extracting some crucial data for the next steps. For each file we are going to do some measurements (localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, hnr05, hnr15 ,hnr25 ,hnr35 ,hnr38). These measurements are taken with the help of the Parselmouth library, which allows us to use Praat in Python code. Once our dataset is built, we are going to export it to a CSV file for later use purpose. After creating the dataset, we are going to initialize a LinearRegresion object onto which we will give the previously processed data to learn. After the training is done, a dump of the LinearRegresion is saved for feature usage. In order to predict a result we just have to instatiate the `DatasetCreator` class and call the `DatasetCreator.predict(wav_path='./path/to/file')` method.

 ```datasets```
 Is the folder containing all the required datasets for the algo to work. We have the following:
 1. `dataset_voices` wav files of healty and parkinson's people. Will be used in the linear regression for the "learning" part
 2. `dataset_our_voices` was used only for testing purposes


