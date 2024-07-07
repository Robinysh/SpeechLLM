install:
	git config core.hooksPath .githooks
	- git submodule update --init --recursive --remote --force
	CFLAGS="-fPIC" CXX_FLAGS="-fPIC" pdm sync --no-isolation -G cuda

install-hpu:
	git config core.hooksPath .githooks
	- git submodule update --init --recursive --remote --force
	CFLAGS="-fPIC" CXX_FLAGS="-fPIC" pdm sync --no-isolation -G hpu
	HABANALABS_VIRTUAL_DIR='.venv' habanalabs-installer.sh install --type pytorch --venv -y
	pdm run pip uninstall mpi4py -y

reset:
	git fetch
	git remote prune origin
	git checkout main -f
	git reset --hard origin/main
	git checkout dev -f
	git reset --hard origin/dev
	$(MAKE) install
