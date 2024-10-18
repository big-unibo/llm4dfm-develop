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
- `pipeline.py -- contains the whole process of importing, batching, querying and storing metrics`
- `metrics.py -- contains the process of calculating metrics`
- `preprocess.py -- contains the preprocessing phase`
- `utils.py    -- contains general utils`
- `csv_graph.py    -- contains the process which generates graphs`
- `graph_utils.py    -- contains utils used to work with graph, such as metrics calculation`
- `.env        -- contains information about program's paths`

Configuration files, script to automate run are collected in `src/resources` module.

- `pipeline-config.yml  -- contains configuration of the run`
- `metrics-config.yml -- contains configuration of metrics`
- `preprocess.yml -- contains preprocessing rules to apply`
- `credentials.yml  -- contains configuration of the second step`
- `csv-graph-config.yml  -- contains configuration of csv-based graph generation`
- `automatic-run.sh  -- script to automate runs`
- `automatic-metrics.sh  -- script to automate metrics script`
- `yml.html  -- script to compare ground-truth and output via visualisation`

## Installation

All Java/Scala dependencies must be managed through Gradle (`build.gradle`). See [here](https://docs.gradle.org/current/userguide/core_dependency_management.html).

> Software projects rarely work in isolation. In most cases, a project relies on reusable functionality in the form of libraries or is broken up into individual components to compose a modularized system. Dependency management is a technique for declaring, resolving and using dependencies required by the project in an automated fashion. Gradle has built-in support for dependency management and lives up to the task of fulfilling typical scenarios encountered in modern software projects. 

All Python dependencies must be managed through virtual environments. See [here](https://docs.python.org/3/library/venv.html).

> The venv module provides support for creating lightweight "virtual environments" with their own site directories, optionally isolated from system site directories. Each virtual environment has its own Python binary (which matches the version of the binary that was used to create this environment) and can have its own independent set of installed Python packages in its site directories.

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
- `AUTO_OUTPUTS    -- path to folder in which outpust of automatic runs are stored`
- `RESULTS   -- path to folder in which results are stored`
- `INPUTS   -- path to folder containing exercise prompts`
- `SAVE_MODELS   -- path to folder in which store imported models`

If using Azure to interact with model's API, these configurations must be provided too (model_name must be uppercase)
- `ENDPOINT_{model-name}`

### Authentication key

Authentication key must be stored in `src/main/resources/credentials.yml`,
an example of how the config is structured can be found in `src/main/resources/credentials-example.yml`.

### Algorithmic parameters

#### Pipeline

The following parameters can be configured in `src/main/resources/pipeline-config.yml` file.

#### Note

**Import model has been momentarily removed**

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
- `deployment     -- Deployment name for azure distribution [test-gpt-35, test-gpt-4o]`
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
- `prompt_version -- the prompt version (part between prompts- and .yml)`

General

- `use -- the model to use between import and api`
- `debug_prints   -- enable output prints during execution`

Output

- `dir_label -- the label used in output directory name`

#### CSV-Graph

The following parameters can be configured in `src/main/resources/csv-graph-config.yml` file.

- `v -- the exercise version (part between exercise-N- and text.yml) [sql, original, demand]`
- `prompt_v -- the prompt version (part between prompts-v and .yml)`
- `model_label -- model's label name`
- `dir_label -- directory in which store file name`

#### Metrics

The following parameters can be configured in `src/main/resources/metrics-config.yml` file, under the `exercise` section.

- `dir -- the exercise's directory inside outputs folder`
- `name -- the exercise name without .yml extension`
- `demand -- whether it's a demand driven exercise [true, false]`
- `gt -- the ground truth's exercise`

## Usage

### Single run

- Setup [authentication](#authentication-key)
- Configure [pipeline](#Pipeline), and [graph](#CSV-Graph)
- Run `python pipeline/pipeline.py` from `src/main/python/` directory.
  If no Exceptions raised, in `outputs` directory a new directory with a file `/{exercise-version}-{exercise-prompt_version}-{model-label}-{dir_label}/{exercise.name}-{exercise.version}-{exercise.prompt_version}-{model.label}-{new_timestamp}.yml` is generated. Its structure is as follows:
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
        name: FACT_NAME
      measures: 
      - name: MEASURE1_NAME
      - name: MEASURE2_NAME
      dependencies:
      - from: TABLE1.Attr
      - to: TABLE2.Attr
      - ...
    - fact:
      name: FACT_NAME
      measures:
      - name: MEASURE1_NAME
      - name: MEASURE2_NAME
      dependencies:
      - from: TABLE1.Attr
      - to: TABLE2.Attr
      - ...
  - output_preprocessed:
    - fact:
      name: FACT_NAME
      measures:
      - name: MEASURE1_NAME
      - name: MEASURE2_NAME
      dependencies:
      - from: TABLE1.Attr
        label: tp
        to: TABLE2.Attr
      - ...
      ground_truth_labels:
        dependencies:
        - from: TABLE1.Attr
          label: tp
          to: TABLE2.Attr
        - ...
        fact:
          name: FACT_NAME
        measures:
        - name: MEASURE1_NAME
        - name: MEASURE2_NAME
    - fact:
      name: FACT_NAME
      measures:
      - name: MEASURE1_NAME
      - name: MEASURE2_NAME
      dependencies:
      - from: TABLE1.Attr
        label: tp
        to: TABLE2.Attr
      - ...
      ground_truth_labels:
        dependencies:
        - from: TABLE1.Attr
          label: tp
          to: TABLE2.Attr
        - ...
        fact:
          name: FACT_NAME
        measures:
        - name: MEASURE1_NAME
        - name: MEASURE2_NAME
  - gt_preprocessed:
    - fact:
      name: FACT_NAME
      measures:
      - name: MEASURE1_NAME
      - name: MEASURE2_NAME
      dependencies:
      - from: TABLE1.Attr
      - to: TABLE2.Attr
      - ...
    - fact:
      name: FACT_NAME
      measures:
      - name: MEASURE1_NAME
      - name: MEASURE2_NAME
      dependencies:
      - from: TABLE1.Attr
      - to: TABLE2.Attr
      - ...
  - metrics:
    - edges:
        precision: [0.0 - 1]
        recall: [0.0 - 1]
        f1: [0.0 - 1]
      nodes:
        precision: [0.0 - 1]
        recall: [0.0 - 1]
        f1: [0.0 - 1]
    - edges:
        precision: [0.0 - 1]
        recall: [0.0 - 1]
        f1: [0.0 - 1]
      nodes:
        precision: [0.0 - 1]
        recall: [0.0 - 1]
        f1: [0.0 - 1]

- Run `python pipeline/csv_graph.py` from `src/main/python/` directory.
  If no Exceptions raised, in `outputs/{csv_graph-v}-{csv_graph-prompt_v}-{csv_graph-model_label}-{csv_graph-dir_label}/` directory a new graph files named `graph-boxplot_f1_edges.pdf, graph-boxplot_f1_nodes.pdf, graph-f1_scores_edges_nodes.pdf, graph-precision_recall_edges.pdf, graph-precision_recall_nodes.pdf` are generated aggregating precision, recall and f1-measure collected in the csv file inside `outputs/{csv_graph-v}-{csv_graph-prompt_v}-{csv_graph-model_label}-{csv_graph-dir_label}/` directory.
Example of run:
`python pipeline/csv_graph.py --exercise_v sql --prompt_version v4 --model gpt4o --runs 1 --label test`
  
- Run `python pipeline/metrics.py` from `src/main/python/` directory.
  If no Exceptions raised, in selected exercise file, metrics section is added/overridden. In case preprocessed ground truth and output are not present, standard ones are used.

### Automatic run

Automatic full step run could be achieved by running `../resources/automatic-run.sh` from `src/main/python/` directory,
after granted execution privileges, by means of `chmod 700 ../resources/automatic-run.sh`.
Run configuration:
- `number_of_runs -- set number of runs, 1 by default`
- `file_version -- set file version [sql, original, demand], sql by default`
- `prompt_version -- set prompt version [v1, v2, v3, v4, demand], v4 by default`
- `model -- model to use in run`
- `"<ex1> ... <fileN>" -- set exercises to run, all files matching previous configurations by default`
- `model_label -- an optional model label used in yml output generated, empty string by default, if empty model name is used`
- `dir_label -- an optional label used in output directory generated, if not provided a timestamp is generated`

All configurations specified as argument **override** the provided by configuration file ones.
If not specified, optional parameters are read by configuration files instead, all **except** dir_label, that in place of automatic run is generated if not given.

Example of run:
`../resources/automatic-run.sh 1 sql v4 gpt "1 2 3 4 5 6 7 8 9" gpt4o test`

Output:
Generate one output file for each run on each file as described before inside `outputs/{file_version}-{prompt_version}-{model_label}-{dir_label}/`.
Additionally, a csv file `output-{file_version}-{prompt_version}-{model_label}.csv` is generated if not present, else is enriched with run output.
Moreover, `pipeline/csv_graph.py` is run too, generating graphs.
