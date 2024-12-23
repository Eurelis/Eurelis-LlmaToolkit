#
# gmake
#
SHELL := /bin/bash
CHDIR_SHELL := $(SHELL)

.ONESHELL: # Applies to every targets in the file!
.SHELLFLAGS += -e # stop at the first failure

PYTHON := python3.11

#
# Setup
#
init-venv:
	@echo "***** $@"
	${PYTHON} -m venv ./.venv

install-pre-commit: init-venv
	@echo "***** $@"
	@source .venv/bin/activate
	pip install pre-commit
	git config --local core.hookspath .githooks/
	chmod +x .githooks/pre-commit

update-venv: init-venv install-pre-commit
	@echo "***** $@"
	@source .venv/bin/activate
	pip install --upgrade pip
	pip install .

install-black: update-venv
	@echo "***** $@"
	@source .venv/bin/activate
	pip install black

install-pylint: update-venv
	@echo "***** $@"
	@source .venv/bin/activate
	pip install pylint

install-mypy: update-venv
	@echo "***** $@"
	@source .venv/bin/activate
	pip install mypy

init-project: update-venv install-black install-pylint install-mypy

check-pre-commit:
	@echo "***** $@"
	pre-commit run --all-files

#
# Build
#
package-build: update-venv
	@echo "***** $@"
	@source .venv/bin/activate
	pip install --upgrade build
	python -m build

package-fast-rebuild:
	@echo "***** $@"
	@source .venv/bin/activate
	python -m build

package-upload: package-build
	@echo "***** $@"
	@source .venv/bin/activate
	pip install --upgrade twine
	twine upload --repository pypi dist/*

#
# Cleaning
#
clean-src:
	@echo "***** $@"
	@source .venv/bin/activate && black src && mypy src

#
# Testing
#
run-tests:
	@echo "***** $@"
	@source .venv/bin/activate
	cd src
	${PYTHON} -m pytest -p no:warnings