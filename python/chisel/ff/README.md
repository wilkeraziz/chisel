# Feature Functions

To add a new feature function, simply implement your idea using this very simple guidelines:

1. Import the interface
	
	```python
	import chisel
	```

2. Configure your extractor

	```python
	@chisel.ff.configure
	def thisMethodConfiguresMyNewFeature(config):
    	"""
	    config is a dictionary containing the strings parsed from chisel's config.ini
    	here you can load stuff into memory (e.g. pre-trained models)
	    """
    	pass
	```

3. Implement your features

	```python
	@chisel.ff.dense
	def MyFeature(hypothesis):
    	"""
	    hypothesis contains the input and the translation
    	this function must return 1 real value
	    in this case the feature will be called 'MyFeature'
    	"""
	    return 0.0
	
	@chisel.ff.features('MyF1', 'MyF2')
	def MyFeatures(hypothesis):
    	"""
	    this function must return 2 real values 
    	in this case 2 features will be computed, they will be named 'MyF1' and 'MyF2', respectively
	    note these are also dense features
    	"""
	    return (0.0, 0.0)

	@chisel.ff.sparse
	def MySparse(hypothesis):
    	"""
	    this function must return a list of named feature values (i.e. pairs of the kind (suffix, fvalue))
    	fetures will be named prefix_suffix, where prefix is the function's name (e.g. MySparse)
	    """
    	return (('v1',0.0) , ('v2', 0.0))
	```

4. If you have a scorer that produces multiple features (perhaps some are dense and some are sparse) which share an underlying computation, you can optimise things quite a bit with `suffstats` and `cleanup`.
	```python
	my_suffstats = None
	
	# this will get called before the scorers
	@chisel.ff.suffstats
	def MyUnderlyingComputation(hypothesis):
		global my_suffstats
		my_suffstats = some_computation(hypothesis)
	
	# this will get called after the scorers
	@chisel.ff.cleanup
	def CleaningAfterMyScorer():
		global my_suffstats
		my_suffstats = None
	```

5. Documentation is important! Thus make sure you start your module with an informative `docstring` telling others how to configure your module and what it produces exactly. If necessary, give some information about the design of your module. Finally, print your `docstring` on the screen in case your module is called as a main program.
    ```python
    if __name__ == '__main__':
        print __doc__
    ```

## Getting help

The idea is that each module is independent, so they should also contain documentation. Well-written modules will give you a nice help message if run from the command line. 

Test the following:

		python -m chisel.ff.klm
