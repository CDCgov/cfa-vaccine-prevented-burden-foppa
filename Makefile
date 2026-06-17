SIMS = output/simulations.parquet
ESTIMATE = output/.estimate.checkpoint
DEMO = output/.demo.checkpoint
COR = output/.cor.checkpoint
QUAL = output/.qual.checkpoint
PACKAGE = burden/__init__.py
CONFIG = scripts/config.json

.PHONY: clean

all: $(COR) $(QUAL) $(DEMO) $(ESTIMATE) $(MS)

$(COR): scripts/correlate.py $(SIMS)
	python $< --output=$@ --input=$(SIMS)

$(ESTIMATE): scripts/estimate.py $(SIMS)
	python $< --output=$@ --input=$(SIMS)

$(DEMO): scripts/demo.py scripts/helpers.py $(PACKAGE)
	python $< --output=$@

$(QUAL): scripts/qualitative.py $(SIMS)
	python $< --input=$(SIMS) --output=$@

$(SIMS): scripts/simulate.py $(PACKAGE) $(CONFIG)
	python $< --output=$@ --config $(CONFIG)

clean:
	rm -f output/* output/.*.checkpoint
