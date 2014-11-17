# MT Evaluation Metrics

MT evaluation metrics can be used to define loss functions used for MBR or Consensus decoding.
To add a new loss function based on your own evaluation metric (decomposable at the sentence level), simply implement the following interface:


1. Import the interface

    ```python
    from chisel.mteval import LossFunction
    ```

2. To configure your metric use the config dictionary passed to the following method

    ```python
    def configure(self, config):
      pass
    ```
    
3. Precompute sufficient statistics from references

    ```python
    def prepare_decoding(self, source, evidence, hypotheses):
      pass
    ```
    
4. Implement the loss against a *reference*

    ```python
    def loss(self, c, r):
      pass
    ```
    

5. Implement the loss against an *expected reference* (aka `consensus loss`)

    ```python
    def coloss(self, c):
      pass
    ```


6. You might want to cleanup stuff after decoding a sentence

    ```python
    def cleanup(self):
      pass
    ```

7. Finally, teach the decoder how to construct the wrapper to your loss function. For that your module must define `construct` which takes an alias and returns an instance of your custom class. Note that this is not where you configure your underlying metric, for that you should be using `configure` in the interface mteval.LossFunction.

        ```python
    def construct(alias):
            return MyWrappedLoss(alias)
    ```


