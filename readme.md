Installing with virtualenv
Take the following steps to install TensorFlow with Virtualenv:

Start a terminal (a shell). You'll perform all subsequent steps in this shell.

Install pip and virtualenv by issuing the following commands:

 $ sudo easy_install pip
 $ pip install --upgrade virtualenv 
Create a virtualenv environment by issuing a command of one of the following formats:

 $ virtualenv --system-site-packages targetDirectory # for Python 2.7
 $ virtualenv --system-site-packages -p python3 targetDirectory # for Python 3.n
 
where targetDirectory identifies the top of the virtualenv tree. Our instructions assume that targetDirectory is ~/tensorflow, but you may choose any directory.

Activate the virtualenv environment by issuing one of the following commands:

$ source ~/tensorflow/bin/activate      # If using bash, sh, ksh, or zsh
$ source ~/tensorflow/bin/activate.csh  # If using csh or tcsh 
The preceding source command should change your prompt to the following:

 (tensorflow)$ 
Ensure pip ≥8.1 is installed:

 (tensorflow)$ easy_install -U pip
Issue one of the following commands to install TensorFlow and all the packages that TensorFlow requires into the active Virtualenv environment:

 (tensorflow)$ pip install --upgrade tensorflow      # for Python 2.7
 (tensorflow)$ pip3 install --upgrade tensorflow     # for Python 3.n

Optional. If Step 6 failed (typically because you invoked a pip version lower than 8.1), install TensorFlow in the active virtualenv environment by issuing a command of the following format:

 $ pip install --upgrade tfBinaryURL   # Python 2.7
 $ pip3 install --upgrade tfBinaryURL  # Python 3.n 
where tfBinaryURL identifies the URL of the TensorFlow Python package. The appropriate value of tfBinaryURL depends on the operating system and Python version. Find the appropriate value for tfBinaryURL for your system here. For example, if you are installing TensorFlow for macOS, Python 2.7, the command to install TensorFlow in the active Virtualenv is as follows:

 $ pip3 install --upgrade \
 https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.4.1-py2-none-any.whl
If you encounter installation problems, see Common Installation Problems.


# Source: https://www.tensorflow.org/install/install_mac#installing_with_virtualenv