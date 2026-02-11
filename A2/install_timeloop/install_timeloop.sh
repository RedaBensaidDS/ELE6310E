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

#TODO: git clone commands assume SSH connection to github is set up.

# ---------------------------------------------------------------------
# CONFIG

## Timeloop installation path
#export TL_INSTALL_PREFIX="/" # for Google Colab
export TL_INSTALL_PREFIX="${HOME}/.local" # for other cases (adjust as necessary)

## Whether to copy previously saved timeloop executables rather than
## recompiling. Use save_timeloop.sh to save the executables and
## libraries once they have been compiled once.
#export TL_USE_SAVED_TIMELOOP=1

## Location where executables and shared libraries can be saved.
## MAKE SURE this path matches the mount point for your drive
export GOOGLE_DRIVE_PATH="/content/gdrive/MyDrive"
export TL_EXEC_SAVE_PATH=${GOOGLE_DRIVE_PATH}/timeloop_colab_executables
# ---------------------------------------------------------------------

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
source ~/install_tl/install_tl_step1.sh
# we should now be in a python virtual environment

echo "---------- STEP 2: Clone accelergy-timeloop-infrastructure -----"
git clone --recurse-submodules https://github.com/Accelergy-Project/accelergy-timeloop-infrastructure.git
cd accelergy-timeloop-infrastructure

echo "---------- STEP 3: Install Accelergy -----------"
# Need to remove accelergy-table-based-plug-ins because it doesn't
# compile. Adding a prefix to the name will exclude it from the build.
mv -v src/{,TMP_}accelergy-table-based-plug-ins
make install_accelergy

# Compile and install Timeloop
echo "---------- STEP 4: Install Timeloop -----------"
if [ $TL_USE_SAVED_TIMELOOP -eq 1 ]
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
		make install_timeloop
		source ~/install_tl/timeloop_make_install.sh
		#TODO: We could directly save the bin and lib at this point if
		#option has been selected instead of requiring a separate script.
fi

echo "---------- STEP 5: Install Timeloop python front-end -----"
source ~/install_tl/install_tl_step5.sh

echo "---------- STEP 6: Retrieve and tweak tutorial -----"
source ~/install_tl/install_tl_step6.sh

# Suggest PATH and LD_LIBRARY_PATH variables
#TODO: suggest update to PATH, unless we are going to always go
#      through the python front-end...
echo "*** Additional step: Ensure shared libs can be found ***"
MSG="export LD_LIBRARY_PATH=\"${TL_INSTALL_PREFIX}/lib"
MSG+=':${LD_LIBRARY_PATH}'
echo $MSG
#TODO: Also need to figure out where barvinok shared libs were installed
