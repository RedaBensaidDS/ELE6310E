TIMELOOP_BASE="${MY_PROJ_DIR}/accelergy-timeloop-infrastructure/src/timeloop"
cd "${MY_PROJ_DIR}/timeloop-python"
export TIMELOOP_INCLUDE_PATH="${TIMELOOP_BASE}/include"
export TIMELOOP_LIB_PATH="${TL_INSTALL_PREFIX}/lib"
rm -Rf build
pip3 install -e .
