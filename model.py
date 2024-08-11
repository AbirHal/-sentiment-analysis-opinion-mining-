
import numpy as np
class ModelPipeline:
    def __init__(self, estimator=None, transformer=None):
        """"
          :transformer, could be a class or any thing, 
                        only it need to have these methods
                         :fit, for learning the transformer
                         :transform, for transforming given input!
          :estimator, our classification model, 
                      or a personalized class for classification, 
                      and it could be anything for classification of given data,
                      it must have these methods,
                         :fit, for learning a estimator
                         :predict, for predicting a given inputs
                         :predict_proba, to provide a probability instead of class
        

        """
        self.transformer = transformer
        self.estimator = estimator
     
        


    def fit(self, X, Y):
        """
        Fit the transformer and the estimator.

        Parameters:
        - X: Training data.
        - Y: Training labels.
        """
        self.transformer.fit(X)
        X_train = self.transformer.transform(X)
        self.estimator.fit(X_train, Y)
    
    def predict(self, X):
        """
        Predict the labels for the given input data.

        Parameters:
        - X: Input data to predict.

        Returns:
        - Predictions for the input data.
        """
        X_test = self.transformer.transform(X)
        return self.estimator.predict(X_test)
    
    def predict_proba(self, X):
        """
        Predict the class probabilities for the given input data.

        Parameters:
        - X: Input data to predict probabilities.

        Returns:
        - Predicted probabilities for the input data.
        """
        X_test = self.transformer.transform(X)
        return self.estimator.predict_proba(X_test)
    
    def get_params(self, deep=True):
        return {
            'estimator': self.estimator,
            'transformer': self.transformer,
           
        }


    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self





