import os
import sys
import matplotlib.pyplot as plt
from facet.auto_emittance import run_automatic_emittance
from ml_tto.automatic_emittance.plotting import plot_screen_profile_measurement

import logging

logging.basicConfig(level=logging.DEBUG)

# set matplotlib logging level to WARNING to suppress debug logs from the plotting library
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# add the path that contains the facet environment
sys.path.insert(0, os.path.join(os.environ["BADGER_RESOURCES"], "facet"))

# add the autonomous control directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from plugins.environments.inj_emit import Environment
from plugins.interfaces.epics import Interface

environment = Environment(interface=Interface())
environment.measure_background = False
environment.save_directory = "."

# remove PVs that are not supported by the VA
for name in list(environment.variables.keys()):
    if (
        "IN10:12" in name
        or "BEND" in name
        or "XCOR" in name
        or "YCOR" in name
        or "KLYS" in name
    ):
        del environment.variables[name]


for val in [-5.0]:
    environment.set_variables({"QUAD:IN10:525:BCTRL": val})
    meas = environment.create_beamprofile_measurement("PROF10711")
    result = meas.measure()
    print(result.rms_sizes)
    fig, ax = plot_screen_profile_measurement(meas)
    fig.savefig(f"test_screen_profile_measurement_{val}.png")

result, fname, X = run_automatic_emittance(environment, "PROF10571")

plt.show()
