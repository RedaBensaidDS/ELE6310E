# Save Timeloop binaries to enable quick install on Google Colab
# Author: Francois Leduc-Primeau <francois.leduc-primeau@polymtl.ca>

# This script relies on environment variables TL_EXEC_SAVE_PATH and
# TL_INSTALL_PREFIX, which should have been set by
# install_timeloop.sh.


# create a directory to store the executables if none exist
mkdir -p ${TL_EXEC_SAVE_PATH}
mkdir -p ${TL_EXEC_SAVE_PATH}/bin
mkdir -p ${TL_EXEC_SAVE_PATH}/lib
# copy the executables
cp -v ${TL_INSTALL_PREFIX}/bin/timeloop-* ${TL_EXEC_SAVE_PATH}/bin/
cp -v ${TL_INSTALL_PREFIX}/bin/einsumgraph ${TL_EXEC_SAVE_PATH}/bin/
cp -v ${TL_INSTALL_PREFIX}/bin/looptree-model ${TL_EXEC_SAVE_PATH}/bin/
# copy the shared libraries
cp -v ${TL_INSTALL_PREFIX}/lib/libtimeloop* ${TL_EXEC_SAVE_PATH}/lib/
cp -v /usr/local/lib/libbarvinok* ${TL_EXEC_SAVE_PATH}/lib/
cp -v /usr/local/lib/libntl* ${TL_EXEC_SAVE_PATH}/lib/
