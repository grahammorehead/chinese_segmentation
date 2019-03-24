# Makefile for Chinese Segmentation	
################################################################################
# Model

construct_vectorizer:
	python vectorizer.py --train

MODEL_SCRIPT = model.py

train_model:
	python $(MODEL_SCRIPT) --train --model_dir models --adaptive_learning_rate

test_model:
	python $(MODEL_SCRIPT) --test --model_dir models

parallel_train:
	python $(MODEL_SCRIPT) --train --model_dir models --adaptive_learning_rate &
	python $(MODEL_SCRIPT) --train --model_dir models --adaptive_learning_rate --skip 20 &
	python $(MODEL_SCRIPT) --train --model_dir models --adaptive_learning_rate --skip 40 &
	python $(MODEL_SCRIPT) --train --model_dir models --adaptive_learning_rate --skip 60 &
	python $(MODEL_SCRIPT) --train --model_dir models --adaptive_learning_rate --skip 80

################################################################################
# Admin

clean:
	# Clean up some artifacts created by other software typically on a Mac
	find . -name Icon$$'\r' -exec rm -fr {} \;
	find . -name '*.pyc' -exec rm -fr {} \;
	find . -name '*~' -exec rm -fr {} \;
	find . -name "*\[Conflict\]*" -exec rm -r {} \;
	find . -name "* \(1\)*" -exec rm -r {} \;
	find . -name ".DS_Store" -exec rm -r {} \;
	find . -name '__pycache__' -exec rm -fr {} \;

force:


share:  force
	rm -fr share
	mkdir share
	cp *.pdf share
	cp *.txt share
	cp *.py share
	mkdir share/data
	cp data/* share/data/
	zip -r morehead.zip share/

################################################################################
################################################################################
