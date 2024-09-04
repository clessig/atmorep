import pathlib
import atmorep.tests.test_utils as test_utils

def pytest_addoption(parser):
    parser.addoption("--field", action="store", help="field to run the test on")
    parser.addoption("--model_id", action="store", help="wandb ID of the atmorep model")
    parser.addoption("--epoch", action="store", help="field to run the test on", default = "0")
    parser.addoption("--strategy", action="store", help="BERT or forecast")
    parser.addoption("--result", action="store", help="result directory path")


def pytest_configure(config):
    """Make parsed options available outside pytest fixtures"""
    
    config_path = pathlib.Path(config.getoption("result"))
    instance = test_utils.ValidationConfig.from_result(config_path)
    test_utils.ValidationConfig.set(instance)
