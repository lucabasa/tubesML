init:
	pip install -r requirements_dev.txt

test:
	coverage run -m pytest tests
    
coverage:
	coverage report -m --include=tubesml/* --skip-covered

release:
	pip install --upgrade setuptools wheel && python setup.py sdist bdist_wheel

publish:
	pip install --upgrade twine && twine upload dist/* --skip-existing
