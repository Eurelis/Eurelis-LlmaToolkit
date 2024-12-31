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
	@uv sync --all-extras

install-pre-commit: init-venv
	@echo "***** $@"
	@git config --local core.hookspath .githooks/
	@chmod +x .githooks/pre-commit

update-venv: init-venv install-pre-commit

init-project: update-venv

update-requirements:
	@echo "***** $@"
	@uv export --quiet --no-header --frozen --no-emit-project --no-hashes --all-extras --no-dev --output-file src/requirements.txt

check-dependancies: update-requirements
	@echo "***** $@"
	@uv tool run pip-audit --requirement src/requirements.txt

check-pre-commit:
	@echo "***** $@"
	@pre-commit run --all-files

#
# Build
#
package-build: update-venv
	@echo "***** $@"
	@source .venv/bin/activate
	uv sync --all-extras
	uv build

package-publish: package-build
	@echo "***** $@"
	@source .venv/bin/activate
	uv publish

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
