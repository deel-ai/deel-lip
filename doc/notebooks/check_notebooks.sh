#!/bin/bash
# jupyter nbconvert --to script $1.ipynb
# python $1.py
# rm $1.py
jupyter nbconvert --to rst --execute --ExecutePreprocessor.timeout=600 $1.ipynb
mv $1.rst ../source/
mv $1_files ../source/
# pastille colab:
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deel-ai/deel-lip/blob/master/chemin_vers_le_notebook.ipynb]
