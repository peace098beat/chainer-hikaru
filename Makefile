.PHONY: test clean release docs

all: test

test:
	# pip install -r test-requirements.txt -q
	# python -m unittest discover -s tests
	python -m unittest discover -s cainer-gogh/tests

clean:
	rm -rf ./log/
	rm -rf ./output/
	rm -rf *.log
	rm -rf *.pyc

clean-pyc:
	rm *.pyc
	rm *.pyo

release:
	python scripts/make-release.py
 

docs:
	$(MAKE) -C docs html
