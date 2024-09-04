from dataclasses import dataclass

"""  @dataclass is used to store data. """

# This DataIngestionArtifact stores the file paths of the training and test datasets.
@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str


# This DataValidationArtifact stores the file paths, validation status and drift report of the training and test datasets.
@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str

# This DataTransformationArtifact stores the file paths of the training and test datasets in the form of numpy array and Transformed object which contains all steps in transformation.
@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str

# This artifact encapsulates the performance metrics of the model, specifically for classification tasks.
@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float


# This artifact contains the path to the trained model file and the associated training and testing metrics.
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact




@dataclass
class ModelEvaluationArtifact:

    is_model_accepted: bool
    improved_accuracy: float
    best_model_path: str
    trained_model_path: str
    train_model_metric_artifact: ClassificationMetricArtifact
    best_model_metric_artifact: ClassificationMetricArtifact



@dataclass
class ModelPusherArtifact:
    saved_model_path:str   
    model_file_path:str