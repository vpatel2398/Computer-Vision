Anaconda
If you want to use just Python and not Colab, I highly recommend installing Anaconda. Anaconda is a necessity for data science and you should learn how to use it because it will change your life.

Anaconda Download

Anaconda Cheat Sheet

Clone GitHub Repo
# Clone
git clone https://github.com/sawyermade/cvfa21netid_project-1.git

# Change directory name to your netid, example using my netid
# You can also use file explorer, right-click, and rename
mv cvfa21netid_project-1 danielsawyer_project-1

# Enter directory, again using my netid as an example
cd danielsawyer_project-1
How To Run: Python
Anaconda Environment Setup
# Install Anaconda Environment
conda env create -f environment.yml
conda activate cvpj1
OR

Pip Setup
# Install Modules Pip
pip3 install -r requirements.txt
Run program
# Runs program, will write output images to output directory
python3 project.py
How To Run: Colab
Open project.py in a text editor and then copy/paste into a new Colab Notebook.

DO NOT ADD ANY NEW CODE CELLS OR TEXT CELLS!!!

Also, do not use any inline "!" bash commands in your code. You wont need it and it will mess up the grading process, which will lose you points since you cant follow instructions.

Once coding is complete go to the File tab in Colab, then Download, choose ".py", save as project.py, and overwrite the original project.py in the original directory you cloned/downloaded earlier. Then submit according to the section below.
