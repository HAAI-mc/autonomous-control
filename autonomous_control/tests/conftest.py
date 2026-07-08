import pytest

import epics
import time

def try_reset_va():
    """
    Try to reset the VA by setting the "RESET" variable to 1. 
    If the operation fails (e.g., if the VA is not available), 
    it will catch the exception and continue without raising an error. 
    """
    try:
        epics.caput("RESET", 1)
        time.sleep(10.0)  # wait for the VA to reset
    except Exception:
        pass


@pytest.fixture(autouse=True)
def reset_va_before_each_test():
    """Reset the virtual accelerator before every test."""
    try_reset_va()
