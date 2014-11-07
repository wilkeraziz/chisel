# MT Evaluation Metrics

To add a new evaluation metric (decomposable at the sentence level), simply implement the following interface:


1. Import the interface

		```python
		import chisel.mteval as mteval
		```

2. Configure your metric

		```python
		@mteval.configure
		def configure(config):
			pass
		```
		
3. Precompute sufficient statistics from references

		```python
		@mteval.training
		def training_ref_precomp(source, references):
			pass
			
		@mteval.decoding
		def decoding_ref_precomp(source, solutions):
			pass
		```
		
4. Precompute sufficient statistics from predictions

		```python
		@mteval.suffstats
		def pred_precomp(predictions):
			pass
		```