init:
	pip install -r requirements.txt

test:
	py.test tests

release:
	pip install --upgrade setuptools wheel && python setup.py sdist bdist_wheel

publish:
	pip install --upgrade twine && twine upload dist/*