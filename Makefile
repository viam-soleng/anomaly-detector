dist/main: src/main.py
	python3 -m venv .venv
	(. .venv/bin/activate && \
	pip3 install -r requirements.txt; \
	python3 -m PyInstaller --onefile --hidden-import="googleapiclient" --add-data="meta.json:." src/main.py; \
	)
	tar -czvf dist/archive.tar.gz dist/main