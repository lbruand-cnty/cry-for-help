import unittest
import ml_api
import pandas as pd
import torch

class MLAPITest(unittest.TestCase):
    def test_create_features(self):
        mlapi = ml_api.MLApi()
        df = pd.read_csv("../test_data/data.csv", dtype=str)
        df_train = df[df.queue == "train"]
        df_test = df[df.queue == "test"]
        df_unlabeled = df[~df.queue.isin(["train", "test"])]

        result = mlapi.create_features(df_unlabeled['texts'].to_list(), df_train['texts'].to_list(), minword=1)
        print(result)
        self.assertEqual(7, len(result))

    def test_make_feature_vector(self):
        mlapi = ml_api.MLApi()
        text_split = "hello Programmation".split()
        feature_index = {'hello': 0, 'Programmation': 1, 'et': 2, 'développement': 3, 'informatique': 4}
        feature_vector = mlapi.make_feature_vector(text_split, feature_index)
        self.assertEqual(1, len(feature_vector))


    def test_train_model(self):
        feature_index = {'hello': 0, 'Programmation': 1, 'et': 2, 'développement': 3, 'informatique': 4}
        mlapi = ml_api.MLApi(labels_index={"GRAIT": 0, "GRAMC": 1},
                             num_labels=2,
                             vocab_size=len(feature_index))
        mlapi.feature_index = feature_index
        df = pd.read_csv("../test_data/data.csv", dtype=str)
        df_train = df[df.queue == "train"]
        df_test = df[df.queue == "test"]

        mlapi.train_model(training_data=df_train,
                          test_data=df_test,
                          )
        self.assertIsNotNone(mlapi.model)

    def test_low_confidence(self):
        feature_index = {'hello': 0, 'Programmation': 1, 'et': 2, 'développement': 3, 'informatique': 4}
        mlapi = ml_api.MLApi(labels_index={"GRAIT": 0, "GRAMC": 1},
                             num_labels=2,
                             vocab_size=len(feature_index))
        mlapi.feature_index = feature_index
        df = pd.read_csv("../test_data/data.csv", dtype=str)
        df_train = df[df.queue == "train"]
        df_test = df[df.queue == "test"]
        df_unlabeled = df[~df.queue.isin(["train", "test"])]

        mlapi.train_model(training_data=df_train,
                          test_data=df_test,
                          )

        df_result = pd.concat(list(mlapi.get_low_conf_unlabeled(df_unlabeled)))
        self.assertEqual(
            len(df_unlabeled),
            len(df_result))





if __name__ == '__main__':
    unittest.main()
