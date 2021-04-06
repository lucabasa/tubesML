init:
	pip install -r requirements.txt

test:
	coverage run -m pytest tests
    
coverage:
	coverage report -m --include=tubesml/*

release:
	pip install --upgrade setuptools wheel && python setup.py sdist bdist_wheel

publish:
	pip install --upgrade twine && twine upload dist/*