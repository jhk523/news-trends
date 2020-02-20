# News Trends Project

This is the GitHub repository of team Seagram.

## Members

- Jaemin Yoo (leader, jaeminyoo@snu.ac.kr)
- Minyong Cho (chominyong@gmail.com)
- Jung hoon Kim (joseph.junghoon.kim@gmail.com)
- Seong Min Lee (ligi214@snu.ac.kr)

## Overview

This repository consists of two main components:
- `src/python` contains source codes for data preprocessing or ML models.
- `src/website` contains source codes for websites.

Python 3.6 is recommended, although Python 3.7 seems to work well. The required
packages are described at `requirements.txt`. All necessary data files such as
the account information for accessing the MySQL DB should be stored at `data`,
which is not included in the current repository.  

## Usage 

The source codes at `src/python/scripts` are directly executable. For instance,
you can train a sentencepiece model by running `python -m scripts.spm` at the
`src/python` directory. Note that `src/python` should be the root directory of
running the scripts, not `src/python/scripts`, for consistency. 
