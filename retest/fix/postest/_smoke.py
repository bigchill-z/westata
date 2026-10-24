import sys, pathlib
sys.path.insert(0, "retest/lib")
from stata_probe import Stata
with Stata() as st:
    print("version", st.version)
    st.run("sysuse auto, clear")
    print("nobs", st.nobs())
