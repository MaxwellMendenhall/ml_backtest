# Machine Learning Class Guide

To start customizing and overriding the default settings for you machine learning you want to import it. In this guide we will creating a class to use Random Forrest Regression (RFR) for our machine learning.&#x20;

To start we want to import the parent class.

```
from memick.ml_util.ml_worker import MachineLearningWorker
```

### Methods and Var's needed to know

There are 3 functions you need to implement. First one is `feature_engineer(self)`, second is `train(self, x_train, y_train, x_test, y_test)`, and final one is `predict(self, x_train, y_train, x_test, y_test)` along with the `__init__(self, data: pd.DataFrame)` constructor.&#x20;

All the accessible variables are `data`, `model`, `predictions`, `get_rows`, `get_columns`, and `get_target`.

Before the methods are explained, the variables must be understood. The parent machine learning class will have default values for everything in case something is not overriden or used. Here are the values and what they do.

<pre class="language-python"><code class="lang-python"><strong># this is a dataframe that provides all the data
</strong><strong># to be used in feature engineering and training 
</strong><strong>self.data = data
</strong><strong>
</strong><strong># this var is used to train and fir with your 
</strong><strong># desired machine learn module
</strong>self.model = None

# use this var for predictions with the model
self.predictions = None

# used to get the number of rows before trade
# entries, used in training
self.get_rows = 10

# used to get columns wanted in to be used in
# training
self.get_columns = ['Close']

# used to set the Y value in machine learning, 
# or in other words the target
self.get_target = 'DifferenceFromOpen'
</code></pre>

### Building the class

Next step is using these variables in the methods. First up is defining the constructor so you have the data avaible to you to perform youe feature engineering on.

```python
from memick.ml_util.ml_worker import MachineLearningWorker
import pandas as pd


class RandomForestRegressorTrainer(MachineLearningWorker):

        def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        # this is the number of rows before each target you want trained
        self.get_rows = 10
        # these are the columns you want trained, in my case I want the column
        # that is already there and a column I am adding in the feature_engineer()
        # method
        self.get_columns = ['Close', 'SMA']
```

After we defined our constructor we can get to the fun part! Creating extra features so we can get the best possible predictions! After we create extra features we can then just call the train method and the prediction method.

```python
from memick.ml_util.ml_worker import MachineLearningWorker
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd


class RandomForestRegressorTrainer(MachineLearningWorker):

        def __init__(self, data: pd.DataFrame):
                super().__init__(data)
                # this is the number of rows before each target you want trained
                self.get_rows = 10
                # these are the columns you want trained, in my case I want the column
                # that is already there and a column I am adding in the feature_engineer()
                # method
                self.get_columns = ['Close', 'SMA']
        
        def feature_engineer(self):
                # here is where you can add addition columns of features you want to be used in training
                # just make sure you edit the 'self.data' with the features you want as that is the dataframe being
                # used in training
                self.data['SMA'] = self.data['Close'].rolling(window=10).mean()
                
        def train(self, x_train, y_train, x_test, y_test):
                # here is where you define the model you want for training
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
                self.model.fit(x_train, y_train)
                
        def predict(self, x_train, y_train, x_test, y_test):
                # here is where the predictions will appear
                # you can get different values for the predictions
                # like calculating the residuals
                self.predictions = self.model.predict(x_test)
        
                mse = mean_squared_error(y_test, self.predictions)
                print(f"Mean Squared Error: {mse}")
```

That is it! The goal of this child class is to take the abstraction of tedous stock market machine learning away from the user. Do not have to worry about how to get the data, just need to define what you want.&#x20;

### Changing Target Warning

The target right now is defined as `self.get_target = 'DifferenceFromOpen'` . Right now the code does not support the change of target.&#x20;
