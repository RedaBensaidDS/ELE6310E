# /usr/bin
cp -v ${TL_SAVE_PATH}/usr/bin/looptree-model /usr/bin/
cp -v ${TL_SAVE_PATH}/usr/bin/timeloop* /usr/bin/
# /usr/lib
cp -v ${TL_SAVE_PATH}/usr/lib/libtimeloop* /usr/lib/
# /usr/local/bin
cp -v ${TL_SAVE_PATH}/usr/local/bin/barvinok* /usr/local/bin/
cp -v ${TL_SAVE_PATH}/usr/local/bin/c2p /usr/local/bin/
cp -v ${TL_SAVE_PATH}/usr/local/bin/disjoint_union* /usr/local/bin/
cp -v ${TL_SAVE_PATH}/usr/local/bin/ehrhart* /usr/local/bin/
cp -v ${TL_SAVE_PATH}/usr/local/bin/findv /usr/local/bin/
cp -v ${TL_SAVE_PATH}/usr/local/bin/iscc /usr/local/bin/
cp -v ${TL_SAVE_PATH}/usr/local/bin/polytope_scan /usr/local/bin/
cp -v ${TL_SAVE_PATH}/usr/local/bin/ppgmp /usr/local/bin/
cp -v ${TL_SAVE_PATH}/usr/local/bin/r2p /usr/local/bin/
cp -v ${TL_SAVE_PATH}/usr/local/bin/timeloop /usr/local/bin/
cp -v ${TL_SAVE_PATH}/usr/local/bin/tl /usr/local/bin/
# /usr/local/lib
cp -v ${TL_SAVE_PATH}/usr/local/lib/libbarvinok* /usr/local/lib/
cp -v ${TL_SAVE_PATH}/usr/local/lib/libisl* /usr/local/lib/
cp -v ${TL_SAVE_PATH}/usr/local/lib/libntl* /usr/local/lib/
cp -v ${TL_SAVE_PATH}/usr/local/lib/libpoly* /usr/local/lib/
# /usr/local/include
cp -v -R ${TL_SAVE_PATH}/usr/local/include/barvinok /usr/local/include/
cp -v -R ${TL_SAVE_PATH}/usr/local/include/isl /usr/local/include/
cp -v -R ${TL_SAVE_PATH}/usr/local/include/NTL /usr/local/include/
cp -v -R ${TL_SAVE_PATH}/usr/local/include/polylib /usr/local/include/

# Set x permission on executables
chmod u+x /usr/bin/looptree-model /usr/bin/timeloop* /usr/local/bin/{barvinok*,c2p,disjoint_union*,ehrhart*,findv,iscc,polytope_scan,ppgmp,r2p,timeloop,tl}
