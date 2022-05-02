import glob
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import time
import joblib


LABELS = [
    "Parkinson", "Jitter_rel", "Jitter_abs", "Jitter_RAP", "Jitter_PPQ", "Shim_loc", "Shim_dB", "Shim_APQ3",
    "Shim_APQ5", "Shi_APQ11", "hnr05", "hnr15", "hnr25", "hnr35", "hnr38"
]
PREDICTORS = [
    "Jitter_rel", "Jitter_abs", "Jitter_RAP", "Jitter_PPQ", "Shim_loc", "Shim_dB", "Shim_APQ3", "Shim_APQ5",
    "Shi_APQ11", "hnr05", "hnr15", "hnr25", "hnr35", "hnr38"
]
CLASS_PARAMS = [
    'file_list', 'localJitter_list', 'local_absolute_jitter_list', 'rapJitter_list', 'ppq5Jitter_list',
    'local_absolute_jitter_list', 'localShimmer_list', 'localdbShimmer_list', 'apq3Shimmer_list', 'aqpq5Shimmer_list',
    'apq11Shimmer_list', 'hnr05_list', 'hnr15_list', 'hnr25_list', 'hnr35_list', 'hnr38_list', 'parkinson_list'
]
UNKNOWN = -1
HEALTHY = 0
PARKINSON = 1


class DatasetCreator:
    """
    This class is responsible with creating a trained model of a liner regression.
    First step is to get the vocal data from a dataset of vocal records. Healthy and Parkinson's ones, in the process
    creating a cvs containing the vocal measurement of the registration and marking which ones are healthy and which
    ones are not
    """
    def __init__(self):
        # If we already have created the measurements and saved them to file, load them and skip the process of creating
        try:
            parkinson = pd.read_csv("algo/dataset/processed_results.csv")
            self.measurements_prediction = parkinson
        except FileNotFoundError:
            self.measurements_prediction = None

        try:
            model = joblib.load("algo/dataset/trained_model.sav")
            self.trained_model = model
        except FileNotFoundError:
            self.trained_model = None

        # Class params for obtaining the csv
        self.file_list = None
        self.localJitter_list = None
        self.local_absolute_jitter_list = None
        self.rapJitter_list = None
        self.ppq5Jitter_list = None
        self.localShimmer_list = None
        self.localdbShimmer_list = None
        self.apq3Shimmer_list = None
        self.aqpq5Shimmer_list = None
        self.apq11Shimmer_list = None
        self.hnr05_list = None
        self.hnr15_list = None
        self.hnr25_list = None
        self.hnr35_list = None
        self.hnr38_list = None
        self.parkinson_list = None

        self.initialize_class_params()
        self.initialize()

    def initialize_class_params(self):
        for label in CLASS_PARAMS:
            setattr(self, label, [])

    def initialize(self):
        initial_time = time.time()
        start_time = time.time()
        print('--- START EXEC ---')

        if self.measurements_prediction is None:
            print('--- START to parse dataset of audios ---')

            self.process_recs()

            print("--- END to parse dataset of audios and generated csv for training data ---", time.time() - start_time)

            start_time = time.time()
            print('--- START to clean data ---')

            self.clean_measurements()

            print("--- END to clean data ---", time.time() - start_time)

        if self.trained_model is None:
            start_time = time.time()
            print('--- START training ---')

            self.train_model()

            print("--- END training ---", time.time() - start_time)

        print("--- END EXEC ---", time.time() - initial_time)

    def append_data_to_lists(self, wave_file, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer,
                             localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, hnr05, hnr15 ,hnr25, hnr35, hnr38,
                             healthy):
        self.file_list.append(wave_file)
        self.localJitter_list.append(localJitter)
        self.local_absolute_jitter_list.append(localabsoluteJitter)
        self.rapJitter_list.append(rapJitter)
        self.ppq5Jitter_list.append(ppq5Jitter)
        self.localShimmer_list.append(localShimmer)
        self.localdbShimmer_list.append(localdbShimmer)
        self.apq3Shimmer_list.append(apq3Shimmer)
        self.aqpq5Shimmer_list.append(aqpq5Shimmer)
        self.apq11Shimmer_list.append(apq11Shimmer)
        self.hnr05_list.append(hnr05)
        self.hnr15_list.append(hnr15)
        self.hnr25_list.append(hnr25)
        self.hnr35_list.append(hnr35)
        self.hnr38_list.append(hnr38)
        self.parkinson_list.append(healthy)

    def process(self, wave_file, healthy):
        sound = parselmouth.Sound(wave_file)
        (
            localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer, localdbShimmer, apq3Shimmer,
            aqpq5Shimmer, apq11Shimmer, hnr05, hnr15, hnr25, hnr35, hnr38
        ) = self.measurements(sound, 75, 1000, "Hertz")
        self.append_data_to_lists(
            wave_file, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer, localdbShimmer,
            apq3Shimmer, aqpq5Shimmer, apq11Shimmer, hnr05, hnr15, hnr25, hnr35, hnr38, healthy
        )

    def process_recs(self):
        start_time = time.time()

        for wave_file in glob.glob("dataset/dataset_voices/spontaneous_dialogue/parkinson/*.wav"):
            self.process(wave_file, PARKINSON)
        print("1. dataset/dataset_voices/spontaneous_dialogue/parkinson/ took", time.time() - start_time, " to load")
        start_time = time.time()

        for wave_file in glob.glob("dataset/dataset_voices/read_text/parkinson/*.wav"):
            self.process(wave_file, PARKINSON)
        print("2. dataset/dataset_voices/read_text/parkinson/ took", time.time() - start_time, " to load")
        start_time = time.time()

        for wave_file in glob.glob("dataset/dataset_voices/spontaneous_dialogue/healthy/*.wav"):
            self.process(wave_file, HEALTHY)
        print("3. dataset/dataset_voices/spontaneous_dialogue/healthy/ took", time.time() - start_time, " to load")
        start_time = time.time()

        for wave_file in glob.glob("dataset/dataset_voices/read_text/healthy/*.wav"):
            self.process(wave_file, HEALTHY)
        print("4. dataset/dataset_voices/read_text/healthy/ took", time.time() - start_time, " to load")

        pred = pd.DataFrame(
            np.column_stack(
                [
                    self.parkinson_list, self.localJitter_list, self.local_absolute_jitter_list, self.rapJitter_list,
                    self.ppq5Jitter_list, self.localShimmer_list, self.localdbShimmer_list, self.apq3Shimmer_list,
                    self.aqpq5Shimmer_list, self.apq11Shimmer_list, self.hnr05_list, self.hnr15_list, self.hnr25_list,
                    self.hnr35_list, self.hnr38_list
                ]
            ),
            columns=[
                "Parkinson", "Jitter_rel", "Jitter_abs", "Jitter_RAP", "Jitter_PPQ", "Shim_loc", "Shim_dB", "Shim_APQ3",
                "Shim_APQ5", "Shi_APQ11", "hnr05", "hnr15", "hnr25", "hnr35", "hnr38"
            ]
        )

        pred.to_csv("dataset/processed_results.csv", index=False)
        self.measurements_prediction = pred

    @staticmethod
    def measurements(voiceID, f0min, f0max, unit="Hertz"):
        """
        Takes as params a voice file and other params and return all the required  measurements done on the sound.
        :param voiceID: The sound file .wav
        :param f0min: the minimum frequency used for reading the sound
        :param f0max: the maximum frequency used for reading the sound
        :param unit: The unit in which the measurements are done default "Hertz"
        :return: A tuple containing this required params localJitter, localabsoluteJitter, rapJitter, ppq5Jitter,
        localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, hnr05, hnr15 ,hnr25 ,hnr35 ,hnr38
        """
        # Read the sound
        sound = parselmouth.Sound(voiceID)

        # Create a praat pitch object
        pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

        localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq11Shimmer = call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harmonicity05 = call(sound, "To Harmonicity (cc)", 0.01, 500, 0.1, 1.0)
        hnr05 = call(harmonicity05, "Get mean", 0, 0)
        harmonicity15 = call(sound, "To Harmonicity (cc)", 0.01, 1500, 0.1, 1.0)
        hnr15 = call(harmonicity15, "Get mean", 0, 0)
        harmonicity25 = call(sound, "To Harmonicity (cc)", 0.01, 2500, 0.1, 1.0)
        hnr25 = call(harmonicity25, "Get mean", 0, 0)
        harmonicity35 = call(sound, "To Harmonicity (cc)", 0.01, 3500, 0.1, 1.0)
        hnr35 = call(harmonicity35, "Get mean", 0, 0)
        harmonicity38 = call(sound, "To Harmonicity (cc)", 0.01, 3800, 0.1, 1.0)
        hnr38 = call(harmonicity38, "Get mean", 0, 0)
        return localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, hnr05, hnr15, hnr25, hnr35, hnr38

    def clean_measurements(self):
        print('Printing the head of the measurements')
        self.measurements_prediction.head()

        for label in LABELS:
            self.measurements_prediction[label].fillna((self.measurements_prediction[label].mean()), inplace=True)

        print('Printing the head of the measurements after cleaning')
        self.measurements_prediction.head()

    def train_model(self):
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression

        x_train, x_test, y_train, y_test = train_test_split(
            self.measurements_prediction[PREDICTORS],
            self.measurements_prediction['Parkinson'],
            test_size=None,
            random_state=2
        )
        print("Slit the data for training and testing")
        print("train shape", x_train.shape, y_train.shape)
        print("test shape", x_test.shape, y_test.shape)

        # Initialize a logistic regression model
        print("Train the Linear Regression model")
        logistic = LogisticRegression()
        logistic.fit(x_train, y_train)

        # Test and train accuracy
        print("Check the accuracy")
        train_score = logistic.score(x_train, y_train)
        test_score = logistic.score(x_test, y_test)
        print('train accuracy =', train_score)
        print('test accuracy =', test_score)

        joblib.dump(logistic, "dataset/trained_model.sav")
        self.trained_model = logistic

    def predict(self, wav_path=None):
        try:
            self.initialize_class_params()
            self.process(wave_file=wav_path, healthy=UNKNOWN)
            to_predict = pd.DataFrame(
                np.column_stack(
                    [
                        self.localJitter_list, self.local_absolute_jitter_list, self.rapJitter_list,
                        self.ppq5Jitter_list, self.localShimmer_list, self.localdbShimmer_list, self.apq3Shimmer_list,
                        self.aqpq5Shimmer_list, self.apq11Shimmer_list, self.hnr05_list, self.hnr15_list,
                        self.hnr25_list, self.hnr35_list, self.hnr38_list
                    ]
                ),
                columns=PREDICTORS
            )
            resp = self.trained_model.predict(to_predict)
            resp = str(resp)

            return {
                "success": True if resp == "[1.]" else False,
                "data": resp
            }

        except Exception as e:
            return {
                "success": False,
                "data": 'jjjj'
            }

# if __name__ == '__main__':
#     dataSet = DatasetCreator()
#     response = dataSet.predict(wav_path="/Users/galoscar07/Documents/master2k20-2k22/4th Semester/algo_detect_parkinson/dataset/test_oscar2.wav")
#     print(response['success'])
#     print(response['data'])
