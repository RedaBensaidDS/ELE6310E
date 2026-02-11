cd .. # i.e. ~/timeloop-accelergy
git clone git@github.com:Accelergy-Project/timeloop-accelergy-exercises.git
# Replace some arch config files
EXERCISE_DIR="timeloop-accelergy-exercises/workspace/tutorial_exercises/01_accelergy_timeloop_2020_ispass/timeloop"
cp -v ~/install_tl/3level.arch.yaml $EXERCISE_DIR/03-model-conv1d+oc-3level/arch/
cp -v ~/install_tl/3level.arch.yaml $EXERCISE_DIR/05-mapper-conv1d+oc-3level/arch/
cp -v ~/install_tl/3levelspatial.arch.yaml $EXERCISE_DIR/04-model-conv1d+oc-3levelspatial/arch/
cp -v ~/install_tl/eyeriss_like.yaml $EXERCISE_DIR/06-mapper-convlayer-eyeriss/arch/
