## Reducing Carbon Emissions of Code Generation in Large Language Models with Line-level Completions

### Repository Structure

- `data/`: Stores all experiment data, including the BigCodeBench dataset processed into the required format.
- `mcs_results/`: Contains MCS results for line-level completions used to calculate the expected test pass rate.
- `models/`: Stores the models used in the experiment.
- `plots/`: Contains code for generating all plots.
- `results/`: Stores generation results, organized by problem, model size, and completion granularity.
- `test_results/`: Holds test results for function-level and line-level completions.
- `tests/`: Contains scripts and resources for executing MCS to calculate the expected test rate and automate restarts.

### Repository Files

- `classes.py`: Contains data classes to store the results.
- `data_processor.py`: Processes the dataset and creates all prompts.
- `experiment.py`: Runs the experiment and performs completions for all prompts.
- `llm.py`: Simplifies interaction with the LLMs used in the experiment.
- `main.py`: Main file to run the complete experiment, combining all components.
- `pyenergibridge_config.json`: Configuration file for PyEnergiBridge. For more information, click [here](https://github.com/luiscruz/pyEnergiBridge).

### Running the Experiment

1. Copy the models to the `models/` folder.
2. Update the model paths in `main.py`.
3. Run `pip install -r requirements.txt`.
4. Run `python3 main.py`.

### Running the Test Pass Rate Calculations

1. Go to the `tests/` folder.
2. Update the local paths used in `monte_carlo_simulation.py`.
3. Run `./monte_carlo_simulation.sh`.

