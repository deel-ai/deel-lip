.PHONY: help prepare-dev test test-disable-gpu doc ipynb-to-rst
.DEFAULT: help

help:
	@echo "make prepare-dev"
	@echo "       create and prepare development environment, use only once"
	@echo "make test"
	@echo "       run tests and linting on py36, py37, py38"
	@echo "make test-disable-gpu"
	@echo "       run test with gpu disabled"
	@echo "make doc"
	@echo "       build Mkdocs documentation"
	@echo "make serve-doc"
	@echo "       run documentation server for development"

prepare-dev:
	python3 -m pip install virtualenv
	python3 -m venv deel_lip_dev_env
	. deel_lip_dev_env/bin/activate && pip install --upgrade pip
	. deel_lip_dev_env/bin/activate && pip install -e .
	. deel_lip_dev_env/bin/activate && pip install -e .[dev]
	. deel_lip_dev_env/bin/activate && pip install -e .[docs]

test:
	. deel_lip_dev_env/bin/activate && tox -e py37-tf23
	. deel_lip_dev_env/bin/activate && tox -e py39-tf27
	. deel_lip_dev_env/bin/activate && tox -e py310-tflatest
	. deel_lip_dev_env/bin/activate && tox -e py310-lint

test-disable-gpu:
	. deel_lip_dev_env/bin/activate && CUDA_VISIBLE_DEVICES=-1 tox -e py37-tf23
	. deel_lip_dev_env/bin/activate && CUDA_VISIBLE_DEVICES=-1 tox -e py39-tf27
	. deel_lip_dev_env/bin/activate && CUDA_VISIBLE_DEVICES=-1 tox -e py310-tflatest
	. deel_lip_dev_env/bin/activate && CUDA_VISIBLE_DEVICES=-1 tox -e py310-lint

doc:
	. deel_lip_dev_env/bin/activate && mkdocs build

serve-doc:
	. deel_lip_dev_env/bin/activate && CUDA_VISIBLE_DEVICES=-1 mkdocs serve
