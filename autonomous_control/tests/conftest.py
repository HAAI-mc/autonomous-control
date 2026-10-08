import epics
import pytest


@pytest.fixture(autouse=True)
def _reset_va_after_test():
    """Reset the virtual accelerator after every test so state doesn't leak between tests."""
    yield
    epics.caput("RESET", 1, wait=True)
