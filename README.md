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

### Guidelines

#### Code

- Add as many *useful* comments as possible
- Delete all *useless* code / resources
- Test *early*, Test *often*, Test *everything* you can
- Write a proper `README.md` (i.e., override this one) that explains:
    - the project structure
    - the algorithmic parameters
    - how to run the project
- Check the output of `./gradlew` to look for warnings (especially in code style)


##### Project structure

- The pipeline 


#### Dataset conventions

- All datasets must be named as follows: `ProjectName-par1_val1-...-parN_valN.csv`
- The only exception is for hive tables: `ProjectName__par1_val1__...__parN_valN.csv`
    - All Spark applications *must* read/write from/to `.csv` files as well as Hive tables
- Schemas for trajectory databases: `(userid, trajectoryid, latitude, longitude, timestamp)` where `timestamp` is unix timestamp (i.e., seconds since 01/01/1970)
    - In `src/main/python/sample.py` you can find an example to transform an uncompliant dataset schema

### Dependency management

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
