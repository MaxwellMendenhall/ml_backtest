import pandas as pd


class MachineLearningInterface:

    def __init__(self, data: pd.DataFrame):

        if type(self).get_model != MachineLearningInterface.get_model:
            raise TypeError("get_model method should not be overridden")

        if isinstance(data, pd.DataFrame):
            self.data = data
            self.model = None
            self.predictions = None
            self.get_rows = 10
            self.get_columns = ['Close']

        else:
            print('Data being passed into MachineLearningWorker is not a list of type DataContainer.')

    def feature_engineer(self):
        """
        Create new features or change current ones. Method will be called
        before training and predict. Make sure to do all desired changes in here.

        :return:
        """
        raise NotImplementedError("This method must be implemented by a subclass")

    def train(self, x_train, y_train, x_test, y_test):
        """
        Will be called after feature engineer.
        Define desired model in here. Fit model in here also.

        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        """
        raise NotImplementedError("This method must be implemented by a subclass")

    def predict(self, x_train, y_train, x_test, y_test):
        """
        Will be called after train. Used to specifying current statistics
        for the trained mode.

        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        """
        raise NotImplementedError("This method must be implemented by a subclass")

    def get_model(self):
        """
        Returns the model used in training.

        :return: model
        """
        return self.model
