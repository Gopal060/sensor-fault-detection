import os , sys
from sensor.exception import SensorCustomException

class TargetValueMapping:

    def __init__(self):
        self.neg: int = 0
        self.pos: int = 1

    # Converts the instance's attributes into a dictionary format.
    def to_dict(self):
        # Returns dict as {"neg": 0, "pos": 1}
        return self.__dict__


    def reverse_mapping(self):
        """ Reverse mapping helps to easily translate the model's output back to the original labels, for easier to understand the results. """
        mapping_response = self.to_dict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))



class SensorModel:

    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise e
    
    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            
            return y_hat
        except Exception as e:
            raise e