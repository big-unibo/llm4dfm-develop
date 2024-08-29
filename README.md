# BIG - LLM 4 DFM

[![build](https://github.com/big-unibo/experimental-project/actions/workflows/build.yml/badge.svg)](https://github.com/big-unibo/experimental-project/actions/workflows/build.yml)

## Project structure

    datasets/   -- where datasets/exercises are stored
        *-text.yml              -- input of the exercise
        *-ground-truth.yml      -- expected result
    inputs/     -- where input prompts are stored
        prompts-v*.yml           -- list of prompts by model
    outputs/    -- where generated datasets are stored (should not be committed)
    results/    -- where experiment/thesis results are stored (must be committed)
        *-result-[model].yml    -- obtained result from [model]
    src/        -- source code

All pipeline python files are collected in `src/main/python/pipeline` module.

- `models.py   -- contains model's utils`
- `pipeline.py -- contains the process of importing, batching and calculating metrixs`
- `utils.py    -- contains general utils`
- `visualisation.py -- contains the process of visualization of the output`
- `visualisation_utils.py    -- contains utils used in visualisation`
- `graph_utils.py    -- contains utils used to work with graph, such as metrics calculation`
- `.env        -- contains information about program's paths`

Configuration files, script to automate run are collected in `src/resources` module.

- `pipeline-config.yml  -- contains configuration of the run`
- `visualisation-config.yml  -- contains configuration of the second step`
- `automatic-run.sh  -- script to automate runs`
- `yml.html  -- script to compare ground-truth and output via visualisation`

## Installation

All Java/Scala dependencies must be managed through Gradle (`build.gradle`). See [here](https://docs.gradle.org/current/userguide/core_dependency_management.html).

> Software projects rarely work in isolation. In most cases, a project relies on reusable functionality in the form of libraries or is broken up into individual components to compose a modularized system. Dependency management is a technique for declaring, resolving and using dependencies required by the project in an automated fashion. Gradle has built-in support for dependency management and lives up to the task of fulfilling typical scenarios encountered in modern software projects. 

All Python dependencies must be managed through virtual environments. See [here](https://docs.python.org/3/library/venv.html).

> The venv module provides support for creating lightweight “virtual environments” with their own site directories, optionally isolated from system site directories. Each virtual environment has its own Python binary (which matches the version of the binary that was used to create this environment) and can have its own independent set of installed Python packages in its site directories.

    cd src/main/python
    python -m venv venv
    pip install -r requirements.txt

To activate venv in Windows (with bash shell; e.g., git bash)

    source venv/Scripts/activate

To activate venv in Linux

    source venv/bin/activate

## Configuration 


### Required configuration parameters

Inside `pipeline` module, a `.env` file must be provided with following configurations:
- `DATASETS   -- path to folder containing exercise texts`
- `OUTPUTS    -- path to folder in which outpust are stored`
- `RESULTS   -- path to folder in which results are stored`
- `INPUTS   -- path to folder containing exercise prompts`
- `SAVE_MODELS   -- path to folder in which store imported models`

If using Azure to interact with model's API, these configurations must be provided too (model_name must be uppercase)
- `ENDPOINT_{model-name}`
- `DEPLOYMENT_NAME_{model-name}`

### Authentication key

Authentication key must be stored in `src/main/resources/credentials.yml`,
an example of how the config is structured can be found in `src/main/resources/credentials-example.yml`.

### Algorithmic parameters

#### Pipeline

The following parameters can be configured in `src/main/resources/pipeline-config.yml` file.

#### Note

**Import model has been momentarily deleted**

Imported model

- `name -- model's name (can be a generalization, such as llama-2, the exact name is stored in "models.py" file, if not present you must add it there)`
- `tokenizer -name   -- model's tokenizer name, usually the same as the model`
- `temperature -- threshold between 0 and 2 that specifies willing to generate more random answers as growing to 1 *if used do_sample must be true`
- `max_new_tokens -- limit the maximum number of tokens generated in a single call`
- `do_sample -- boolean, if set specifies to generate more creative output`
- `top_p -- threshold between 0 and 1 that specifies willing to use a wider set of words as growing to 1 *if used do_sample must be true`
- `quantization -- boolean, enabling quantization techniques to speed up process slightly reducing accuracy`

Api model

- `name        -- model's name (can be a generalization, such as llama-2, the exact name is stored in "models.py" file, if not present you must add it there)`
- `label     -- model's name used in output file name generated, if not specified it uses name`
- `version     -- model's version if present [actually working only for gpt]`
- `api_version     -- api model's version`
- `max_tokens -- it's the maximum length of the generated output`
- `n_response -- regulates number of responses the model generates`
- `temperature -- threshold between 0 and 2 that specifies willing to generate more random answers as growing to 2 [0, ..., 1] [Default 0.1]`
- `stop        -- set the stop character(s, if list) that terminate the response when encountered`
- `top_p -- threshold between 0 and 1 that specifies willing to use a wider set of words as growing to 1 [0, ..., 1]`
- `top_k -- threshold between 1 and 40 that specifies the number of tokens (with the highest probability) considered for the next generation. Less randomness for lower values [could be only a gemini parameter]`

Exercise

- `name           -- the exercise name (part before version, exercise-N)`
- `version           -- the exercise version (part between exercise-N- and text.yml) [sql, original, demand]`
- `prompt_version -- the prompt version (part between prompts-v and .yml)`

General

- `use -- the model to use between import and api`
- `debug_prints   -- enable output prints during execution`

#### Visualisation

The following parameters can be configured in `src/main/resources/visualisation-config.yml` file.

Exercise
Given that it's required to read both ground-truth and model output, to make it easier to configure, different ways can be used to state the exercise to read.

- `full_name           -- the output exercise full name (part before -text.yml)\n ** If provided, no further options of the exercise have to be passed`
- `name -- the exercise name (exercise-*.*)`
- `v           -- the exercise version (sql, original-text, ...)`
- `prompt_v -- the prompt version of the output exercise (v*)`
- `latest           -- boolean that enable the retrieval of latest timestamp matching previous configurations`
- `timestamp -- if not latest, provide the timestamp in format YYYY-MM-DDTHH-mm_ss`

Model

- `name -- model name `
- `v           -- model version, **use only if present in file name`

Visualization

Configurations which regulate graph visualization.

- `node_color -- boolean, enable node colors (default green if TP, grey if FN, red if FP)`
- `edge_color -- boolean, enable edge colors (default green if TP, grey if FN, red if FP)`
- `arrowsize -- regulates edge's arrow pointer dimension`
- `font_size -- regulates font dimension`
- `node_size -- regulates node dimension`
- `image`
  - `format -- the image export format`
- `show_graph -- boolean, enable graph visualization`
- `dag_graph  -- boolean, if true avoid auto dependency visualization, enabling DAG visualization, and color nodes differently in case of auto dependencies`
- `table_names  -- boolean, if true table names are considered for comparing, and node attributes are shown with table name otherwise they aren't considered and tables names are not shown in DAG`


## Usage

### Single run

- Setup [authentication](#authentication-key)
- Configure [pipeline](#Pipeline) and [visualisation](#Visualisation)
- Run `python pipeline/pipeline.py` from `src/main/python/` directory 
  If no Exceptions raised, in `outputs` directory a new file with name `{exercise.name}-{exercise.version}-{exercise.prompt_version}-{model.label}-{new_timestamp}.yml` is generated. Its structure is as follows:
  - config:
    - name: gpt
    - version: 3.5-turbo
    - max_tokens: null
    - n_responses: 1
    - stop: null
    - top_p: 0.9
    - top_k: 5
  - output:
    - fact:
      - name: FACT_NAME
    - measures:
      - name: MEASURE_NAME
    - dependencies:
      - from: TABLE1.Attr
      - to: TABLE2.Attr
      - ...
  - metrics:
    - edges:
      - precision: [0.0 - 1]
      - recall: [0.0 - 1]
      - f1: [0.0 - 1]
    - nodes:
      - precision: [0.0 - 1]
      - recall: [0.0 - 1]
      - f1: [0.0 - 1]
  
- Run `python pipeline/visualisation.py` from `src/main/python/` directory
  If no Exceptions raised, in `outputs` directory a new file with name `{exercise name matching config}.{visualisation.image.format}` is generated as graph representation, labeling green nodes and edges for true positive, red for false positive and grey for false negative.

### Automatic run

Automatic full step run could be achieved by running `./pipeline/automatic-run.sh` after granted execution privileges,
by means of `chmod 700 ./pipeline/automatic-run.sh`.
Run configuration:
- `number_of_runs -- set number of runs, 1 by default`
- `file_version -- set file version [sql, original, demand], sql by default`
- `prompt_version -- set prompt version [v1, v2, v3, v4, demand], v4 by default`
- `<ex1> ... <fileN> -- set exercises to run, all files matching previous configurations by default`

Example of run:
`./pipeline/automatic-run.sh 1 sql v3 4 1`

Output:
Generate one output file for each run on each file as described before. 

*Not fully working yet* - Additionally, a csv file `outputs/automatic_run/output.csv` is enriched with run configurations, output and metrics.
