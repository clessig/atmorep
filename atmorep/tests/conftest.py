def pytest_addoption(parser):
    parser.addoption("--field", action="store", help="field to run the test on")
    parser.addoption("--model_id", action="store", help="wandb ID of the atmorep model")
    parser.addoption("--epoch", action="store", help="field to run the test on", default = "0")

