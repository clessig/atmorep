import atmorep.tests.test_utils as test_utils

def pytest_addoption(parser):
    parser.addoption("--field", action="store", help="field to run the test on")
    parser.addoption("--model_id", action="store", help="wandb ID of the atmorep model")
    parser.addoption("--epoch", action="store", help="field to run the test on", default = "0")
    parser.addoption("--strategy", action="store", help="BERT or forecast")


def pytest_configure(config):
    """Make parsed options available outside pytest fixtures"""
    
    test_utils.ValidationConfig.set(
        config.getoption("field"),
        config.getoption("model_id"),
        int(config.getoption("epoch")),
        config.getoption("strategy")
    )
