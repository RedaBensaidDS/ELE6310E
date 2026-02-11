cd .. # i.e. ~/timeloop-accelergy
git clone git@github.com:Accelergy-Project/timeloop-accelergy-exercises.git
# Replace some arch config files
EXERCISE_SRC="~/install_tl/tutorial_cfg"
EXERCISE_DEST="timeloop-accelergy-exercises/workspace/tutorial_exercises/01_accelergy_timeloop_2020_ispass/timeloop"
cp -v $EXERCISE_SRC/3level.arch.yaml $EXERCISE_DEST/03-model-conv1d+oc-3level/arch/
cp -v $EXERCISE_SRC/3level.arch.yaml $EXERCISE_DEST/05-mapper-conv1d+oc-3level/arch/
cp -v $EXERCISE_SRC/3levelspatial.arch.yaml $EXERCISE_DEST/04-model-conv1d+oc-3levelspatial/arch/
cp -v $EXERCISE_SRC/eyeriss_like.yaml $EXERCISE_DEST/06-mapper-convlayer-eyeriss/arch/
