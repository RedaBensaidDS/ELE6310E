# Save files installed by Timeloop to enable quick install on Google Colab
# Author: Francois Leduc-Primeau <francois.leduc-primeau@polymtl.ca>

# This script relies on environment variable TL_SAVE_PATH, which
# should have been set by install_timeloop.sh.

# create the directory structure
mkdir -p ${TL_SAVE_PATH}
mkdir -p ${TL_SAVE_PATH}/usr/bin
mkdir -p ${TL_SAVE_PATH}/usr/lib
mkdir -p ${TL_SAVE_PATH}/usr/local/bin
mkdir -p ${TL_SAVE_PATH}/usr/local/lib
mkdir -p ${TL_SAVE_PATH}/usr/local/include

# /usr/bin
cp -v /usr/bin/looptree-model ${TL_SAVE_PATH}/usr/bin/
cp -v /usr/bin/timeloop*      ${TL_SAVE_PATH}/usr/bin/
# /usr/lib
cp -v /usr/lib/libtimeloop* ${TL_SAVE_PATH}/usr/lib/
# /usr/local/bin
cp -v /usr/local/bin/barvinok*       ${TL_SAVE_PATH}/usr/local/bin/
cp -v /usr/local/bin/c2p             ${TL_SAVE_PATH}/usr/local/bin/
cp -v /usr/local/bin/disjoint_union* ${TL_SAVE_PATH}/usr/local/bin/
cp -v /usr/local/bin/ehrhart*        ${TL_SAVE_PATH}/usr/local/bin/
cp -v /usr/local/bin/findv           ${TL_SAVE_PATH}/usr/local/bin/
cp -v /usr/local/bin/iscc            ${TL_SAVE_PATH}/usr/local/bin/
cp -v /usr/local/bin/polytope_scan   ${TL_SAVE_PATH}/usr/local/bin/
cp -v /usr/local/bin/ppgmp           ${TL_SAVE_PATH}/usr/local/bin/
cp -v /usr/local/bin/r2p             ${TL_SAVE_PATH}/usr/local/bin/
cp -v /usr/local/bin/timeloop        ${TL_SAVE_PATH}/usr/local/bin/
cp -v /usr/local/bin/tl              ${TL_SAVE_PATH}/usr/local/bin/
# /usr/local/lib
cp -v /usr/local/lib/libbarvinok* ${TL_SAVE_PATH}/usr/local/lib/
cp -v /usr/local/lib/libisl*      ${TL_SAVE_PATH}/usr/local/lib/
cp -v /usr/local/lib/libntl*      ${TL_SAVE_PATH}/usr/local/lib/
cp -v /usr/local/lib/libpoly*     ${TL_SAVE_PATH}/usr/local/lib/
# /usr/local/include
cp -v -R /usr/local/include/barvinok ${TL_SAVE_PATH}/usr/local/include/
cp -v -R /usr/local/include/isl      ${TL_SAVE_PATH}/usr/local/include/
cp -v -R /usr/local/include/NTL      ${TL_SAVE_PATH}/usr/local/include/
cp -v -R /usr/local/include/polylib  ${TL_SAVE_PATH}/usr/local/include/
