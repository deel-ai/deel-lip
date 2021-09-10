cd notebooks
bash ./check_notebooks.sh demo0
bash ./check_notebooks.sh demo1
bash ./check_notebooks.sh demo2
bash ./check_notebooks.sh demo3
bash ./check_notebooks.sh demo4
cd ..

sphinx-apidoc --implicit-namespaces -f -e -o source ..\deel

make html
