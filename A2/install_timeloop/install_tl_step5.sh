cd .. # i.e. ~/timeloop-accelergy
TIMELOOP_BASE="$PWD"
TIMELOOP_BASE+="/accelergy-timeloop-infrastructure/src/timeloop"
git clone --recurse-submodules git@github.com:Accelergy-Project/timeloop-python.git
cd timeloop-python
export TIMELOOP_INCLUDE_PATH="$TIMELOOP_BASE/include"
export TIMELOOP_LIB_PATH="$TIMELOOP_BASE/build"
#TODO: Would need to remove the build/ directory if it exists
pip3 install -e .
