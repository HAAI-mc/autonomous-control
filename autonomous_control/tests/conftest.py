import pytest

import epics
import time


def _wait_for_va_ready(pv_name: str = "QUAD:IN10:121:BCTRL", timeout_s: int = 60) -> None:
    """Poll *pv_name* until it is connected and returning a value (up to *timeout_s* seconds)."""
    pv = epics.get_pv(pv_name, auto_monitor=False)
    for _ in range(timeout_s):
        if pv.wait_for_connection(timeout=0.5) and pv.get() is not None:
            return
        time.sleep(0.5)


def try_reset_va():
    """
    Try to reset the VA by setting the "RESET" variable to 1.
    After the reset, poll until the VA is ready again (up to 60 s).
    If the operation fails (e.g., if the VA is not available),
    it will catch the exception and continue without raising an error.
    """
    try:
        epics.caput("RESET", 1)
        # Wait for the VA to finish resetting before the next test.
        _wait_for_va_ready(timeout_s=60)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def reset_va_before_each_test():
    """Reset the virtual accelerator before every test."""
    try_reset_va()
