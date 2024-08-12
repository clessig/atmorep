import atmorep.utils.logger as logger
import logging


def test_logger_old_formatting(caplog):
    key, value = "my_key", "my_value"

    caplog.set_level(logging.INFO)
    logger.logger.info("%s,%s", key, value)

    assert f"{key},{value}" in caplog.text


def test_logger_new_formatting(caplog):
    key, value = "my_key", "my_value"

    caplog.set_level(logging.INFO)
    logger.logger.info("{:},{:}", key, value)

    assert f"{key},{value}" in caplog.text
