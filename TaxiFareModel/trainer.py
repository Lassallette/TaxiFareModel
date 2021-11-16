from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import clean_data, get_data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from TaxiFareModel.encoders import DistanceTransformer
from TaxiFareModel.encoders import TimeFeaturesEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        '''returns a pipeline model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        self.pipeline=Pipeline([('preproc', preproc_pipe),
                                ('linear_model', LinearRegression())])

    def run(self,X_train,y_train):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse


if __name__ == "__main__":
    data=get_data()
    data=clean_data(data)
    y = data["fare_amount"]
    X = data.drop("fare_amount", axis=1)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    my_pipe=Trainer(X_train,y_train)
    my_pipe.run(my_pipe.X,my_pipe.y)
    my_pipe.evaluate(my_pipe.X,my_pipe.y)
