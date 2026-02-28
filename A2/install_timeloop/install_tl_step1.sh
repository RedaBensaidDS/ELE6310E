# Create a project directory and a python virtual environment
mkdir -p ${MY_PROJ_DIR}
cd ${MY_PROJ_DIR}
# Note: Not necessary to create a venv on Colab.
if [ "${COLAB_ENV}" != "1" ]; then
		python3 -m venv venv;
		source venv/bin/activate
fi
