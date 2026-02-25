# Installation script for Timeloop-Accelergy on Ubuntu and Google Colab
# Author: Francois Leduc-Primeau <francois.leduc-primeau@polymtl.ca>

# Usage: source <SCRIPT LOCATION>/install_timeloop.sh
# where <SCRIPT LOCATION> can be "." if your current
# directory is the location of this script.

# NOTE: On Google Colab, you probably want to access this script from
# a location in your google drive. Your google drive can be mounted by
# executing the following notebook cell:
### from google.colab import drive
### drive.mount('/content/gdrive')

# ---------------------------------------------------------------------
#                           CONFIG
# ---------------------------------------------------------------------
## Set to 1 if running on Google Colab, 0 otherwise.
export COLAB_ENV=0

## Timeloop installation path
if [ "${COLAB_ENV}" = "1" ]; then
		export TL_INSTALL_PREFIX="" # for Google Colab
else
		export TL_INSTALL_PREFIX="${HOME}/.local" # for other cases (adjust as necessary)
fi

## Number of parallel jobs for make (should set to 8 or higher if
## machine allows it)
JOBS=8

## Whether to copy previously saved timeloop executables rather than
## recompiling. Use save_timeloop.sh to save the executables and
## libraries once they have been compiled once.
#TODO: DOESN'T WORK!
#export TL_USE_SAVED_TIMELOOP=1

## Location where executables and shared libraries can be saved.
## MAKE SURE this path matches the mount point for your drive
export GOOGLE_DRIVE_PATH="/content/gdrive/MyDrive"
export TL_EXEC_SAVE_PATH="${GOOGLE_DRIVE_PATH}/timeloop_colab_executables"

## Can optionally get the git projects from Google Drive instead of
## cloning from github (recommended for Colab). In this case the
## projects listed in Step 2 must first be cloned manually in the
## $PROJ_SRC directory specified below.
PROJ_SRC="${GOOGLE_DRIVE_PATH}/timeloop_git_projects"
if [ "${COLAB_ENV}" = "1" ]; then
		# Value in Colab
		LN_INSTEAD_OF_CLONE=1
else
		# Value elsewhere
		LN_INSTEAD_OF_CLONE=0
fi
# ------------------------END CONFIG----------------------------------

# Get location of this script
# https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )

# Create a symlink in the home directory
test -e ~/install_tl && rm -i ~/install_tl
echo "Creating symlink to ${DIR}"
ln -sv ${DIR} ~/install_tl

# When debugging, we want to execute each step manually, so we stop here.
if [ "${DEBUG_TL_INSTALL}" = "1" ]; then
    echo "DEBUG_TL_INSTALL=1 → stopping"
    return 0
fi

echo "--- STEP 0: Installing system dependencies ---"
cd ~
source ~/install_tl/install_tl_step0.sh

echo "---------- STEP 1: Create project dir and venv --------"
export MY_PROJ_DIR="${PWD}/timeloop-accelergy"
source ~/install_tl/install_tl_step1.sh
# we should now be in a python virtual environment

echo "---------- STEP 2: Clone or symlink all projects -----"
if [ $LN_INSTEAD_OF_CLONE -eq 1 ]
then
		ln -s ${PROJ_SRC}/accelergy-timeloop-infrastructure .
		ln -s ${PROJ_SRC}/timeloop-python .
		ln -s ${PROJ_SRC}/timeloop-accelergy-exercises .
else
		# accelergy-timeloop-infrastructure
		git clone --recurse-submodules https://github.com/Accelergy-Project/accelergy-timeloop-infrastructure.git
		# python front-end: timeloop-python
		git clone --recurse-submodules https://github.com/Accelergy-Project/timeloop-python.git
		# Tutorial
		git clone https://github.com/Accelergy-Project/timeloop-accelergy-exercises.git
fi

echo "---------- STEP 3: Install Accelergy -----------"
cd accelergy-timeloop-infrastructure
python3 -m pip install pyyaml # (missing dep in Makefile)
make install_accelergy

# Compile and install Timeloop
echo "---------- STEP 4: Install Timeloop -----------"
if [ "$TL_USE_SAVED_TIMELOOP" = "1" ];
then
		echo "Installing previously saved Timeloop executables:";
		source ~/install_tl/timeloop_make_install_from_saved.sh
else
		echo "Compiling Timeloop...";
		# Keep only the first 173 lines of the Makefile to remove the
		# hardcoded install paths
		head -n173 Makefile > Makefile_new
		rm Makefile
		mv Makefile_new Makefile
		# also update the number of parallel jobs
		sed -i -E "s/\bmake[[:space:]]+-j[[:space:]]*8\b/make -j${JOBS}/g" Makefile
		sed -i -E "s/\bscons[[:space:]]+-j[[:space:]]*8\b/scons -j${JOBS}/g" Makefile
		make install_timeloop
		source ~/install_tl/timeloop_make_install.sh
fi

echo "---------- STEP 5: Install Timeloop python front-end -----"
source ~/install_tl/install_tl_step5.sh

echo "---------- STEP 6: Retrieve and tweak tutorial -----"
source ~/install_tl/install_tl_step6.sh

# Set/Suggest PATH and LD_LIBRARY_PATH variables
echo "---------- STEP 7: Set PATH and LD_LIBRARY_PATH env variables -----"
source ~/install_tl/install_tl_step7.sh

# Additional tweak on Colab:
if [ "${COLAB_ENV}" = "1" ]; then
		chmod u+x /usr/local/share/accelergy/estimation_plug_ins/accelergy-cacti-plug-in/cacti
fi
