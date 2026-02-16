# This script is expected to be sourced from the
# accelergy-timeloop-infrastructure directory (see
# install_timeloop.sh).

# Copy executables
mkdir -p ${TL_INSTALL_PREFIX}/bin
cp -v ${TL_EXEC_SAVE_PATH}/bin/* ${TL_INSTALL_PREFIX}/bin/
chmod u+x ${TL_INSTALL_PREFIX}/bin/*

# Copy shared libraries
mkdir -p ${TL_INSTALL_PREFIX}/lib
cp -v ${TL_EXEC_SAVE_PATH}/lib/* ${TL_INSTALL_PREFIX}/lib/
