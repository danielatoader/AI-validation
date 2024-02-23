# CS4370 - Testing and Validation for AI-Intensive Systems

Welcome to Testing and Validation for AI-Intensive Systems.

This repository will serve as your starting point for each weeks lab work. It will also help you with getting your environment setup. If you have any questions, please don't hesitate to reach out to any of the staff.

#
### <u>Code Editor</u>
<b>NOTE</b>: <i>If you already have an editor that you're content with, then please disregard this section. Your editor though, should be able to handle python and jupyter files!</i>

----
To get started, you first need to have a way of writing and editing code on your system. The best choice for this is VS Code, because it's free and works well on most Operating Systems.

https://code.visualstudio.com/Download

Once VS Code is installed and loaded you'll also want to add a couple useful extensions.

#### <b>Extensions</b>:
    Python
    Python Extension Pack
    Jupyter
    Synthwave '84 (Optional, makes you environment look awesome!)

Install the above mentioned extensions by clicking the extensions button on the side panel and searching for them.

### <u>Python</u>

What good is a code editor without anything to compile your code. Next we're going to install Python.

https://www.python.org/downloads/

Head to the download page for python, download and install the latest version of python 3.10 and install.

Confirm Python is installed correctly by opening a terminal and typing:

    python --version

This should output the version of Python your system is running.

### <u>Pipenv</u>

To make life easy for everyone, we've setup a pip file to ensure quick and easy install of dependencies.

<b>NOTE</b>: <i>Before installing any dependencies, open the Pipfile in your editor and un-comment the version of Tensorflow for your system.</i>

Open a terminal/ powershell and navigate to the project folder, then type:

    pipenv shell

This will put your current session into the python virtual environment. Then type:

    pipenv install

This will install the dependencies defined in the Pipfile, into this specific environment. By doing this, we can ensure no cross dependency issues when working on different python projects.

---

You will need to enable this virtual environment in your code editor to ensure it uses the correct dependencies. For VS Code, this can be found in the bottom right corner of the UI.

It will currently likely show your current Python version. Click this and it will open up the 'Select Interpreter' drop down. For myself, the environment starts with <b><i>'labs'</i></b>, which I then click on to enable as my interpreter.

Yours will likely be the same, or if different, will be shown in your terminal/ powershell window when you typed 'pipenv shell' before.

That should have you up and running! Enjoy the labs and if you have any issues with this, please reach out to the staff and we'll do our best to get you going.