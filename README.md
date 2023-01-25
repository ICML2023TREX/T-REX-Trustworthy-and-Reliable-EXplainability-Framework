# Explainable AI
This repository contains metrics for model explainability evaluation.

This code get from user model and dataset and evaluate it on different metrics.

### DataModule

Dataset inheritance ExplainabilityDataModule and should contain dataset class, and all data needed to create 
at least training and validation dataset.

### Run
#### ExplainabilityModule
This module take model and other necessary data and run your model on the specified using specified evaluator.

#### Evaluator
This is method, you use to evaluate your model, in example you could see how it works with
captum LayerGradCam method.



### Examples
There are two scripts you could run.

trainer.py -  it's script to train your model on specified DataModule.

runner.py - it's script to evaluate your explanator on specified Dataset.


## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
