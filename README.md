# Project 1: Transformations

## Project Info
[Project 1 Info](Project_1.pdf)

## Anaconda
If you want to use just Python and not Colab, I highly recommend installing Anaconda. Anaconda is a necessity for data science and you should learn how to use it because it will change your life.

[Anaconda Download](https://www.anaconda.com/products/individual)

[Anaconda Cheat Sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)

## Clone GitHub Repo
```
# Clone
git clone https://github.com/sawyermade/cvfa21netid_project-1.git

# Change directory name to your netid, example using my netid
# You can also use file explorer, right-click, and rename
mv cvfa21netid_project-1 danielsawyer_project-1

# Enter directory, again using my netid as an example
cd danielsawyer_project-1
```

## How To Run: Python

### Anaconda Environment Setup
```
# Install Anaconda Environment
conda env create -f environment.yml
conda activate cvpj1
```

OR

### Pip Setup
```
# Install Modules Pip
pip3 install -r requirements.txt
```

### Run program
```
# Runs program, will write output images to output directory
python3 project.py
```

## How To Run: Colab
Open project.py in a text editor and then copy/paste into a new Colab Notebook.

DO NOT ADD ANY NEW CODE CELLS OR TEXT CELLS!!!

Also, do not use any inline "!" bash commands in your code. You wont need it and it will mess up the grading process, which will lose you points since you cant follow instructions.

Once coding is complete go to the File tab in Colab, then Download, choose ".py", save as project.py, and overwrite the original project.py in the original directory you cloned/downloaded earlier. Then submit according to the section below.

## How To Submit
For submission replace the cvfa21netid part of the directory with your netid. In my case, my netid is danielsawyer so the directory name would be danielsawyer_project-1

The whole project should be contained within that directory. Then zip the directory, and only that directory, then save it as netid_project-1.zip where netid is replaced by your netid. In my case, it would be danielsawyer_project-1.zip

DO NOT INCLUDE OUTPUT DIRECTORY!!!

So either delete or move the output directory.

ALSO include your 2 PAGE REPORT as report.pdf in the main directory.

After creating the zip file, upload it to Canvas by the submission due date.

Here is an example tree of the directory structure you should be turning in with the netid being danielsawyer.

danielsawyer_project-1.zip contains...

```
danielsawyer_project-1
├── data
│   ├── batch
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── 3.jpg
│   │   ├── 4.jpg
│   │   ├── 5.jpg
│   │   ├── 6.jpg
│   │   ├── 7.jpg
│   │   ├── 8.jpg
│   │   └── 9.jpg
│   ├── img1054.jpg
│   ├── img1166.jpg
│   ├── img1325.jpg
│   ├── img1329.jpg
│   ├── img1378.jpg
│   ├── img1436.jpg
│   ├── img1488.jpg
│   ├── img1500.jpg
│   ├── img1503.jpg
│   ├── img1504.jpg
│   ├── img1506.jpg
│   ├── img1508.jpg
│   ├── img1513.jpg
│   ├── img1517.jpg
│   ├── img1518.jpg
│   ├── img1524.jpg
│   ├── img1525.jpg
│   ├── img1532.jpg
│   ├── img1534.jpg
│   ├── img286.jpg
│   ├── img517.jpg
│   ├── img747.jpg
│   ├── img748.jpg
│   ├── img788.jpg
│   ├── img807.jpg
│   ├── img813.jpg
│   ├── img820.jpg
│   ├── img833.jpg
│   ├── img837.jpg
│   ├── img850.jpg
│   ├── img852.jpg
│   └── README.png
├── environment.yml
├── Project_1.pdf
├── project.py
├── README.md
├── report.pdf
├── requirements.txt
└── transformations.txt
```
