# Contributing Guidelines

Thanks for your interest in contributing to `laplax`! We welcome contributions from the community to help improve the project. Please follow the guidelines below to ensure a smooth contribution process.

## How to Contribute

1. **Fork the Repository**: Start by forking the `laplax` repository on GitHub to your own account.
2. **Clone Your Fork**: Clone your forked repository to your local machine using:
   ```
   git clone <your-fork-url>
   ```
3. **Install Dependencies**: Navigate to the project directory and install the required dependencies. We use [uv](https://docs.astral.sh/uv/) for managing dependencies, and highly recommend that you do too. To get started quickly, after installing `uv`, you can simply run:
    ```
    uv sync --all-extras
    ```
    and this will create a virtual environment and install all necessary packages for development.

3. **Set up pre-commit hooks**: We use pre-commit hooks to maintain code quality. To initialise pre-commit hooks, run the following command:
   ```
   uv run pre-commit install
   ```
3. **Create a Branch**: Create a new branch for your feature or bug fix:
   ```
   git checkout -b <new-feature-branch>
   ```
4. **Make Changes**: Implement your changes in the codebase. Please ensure that your code includes appropriate tests.
5. **Run Tests**: Before submitting your changes, run the test suite to ensure everything is working correctly:
   ```
   uv run pytest
   ```
5. **Commit Changes**: Commit your changes with a clear and descriptive commit message.
6. **Push Changes**: Push your changes to your forked repository:
   ```
   git push origin <new-feature-branch>
   ```
7. **Create a Pull Request**: Open a pull request in the [original `laplax` repository](https://github.com/laplax-org/laplax).

## Notebooks

We use `jupytext` to manage Jupyer notebooks, storing them as `.py` files for better version control. In  order to convert between `.ipynb` and `.py` formats, you can use the following commands:

- To convert a `.ipynb` file to a `.py` file:
  ```
  jupytext --to py:percent <notebook>.ipynb
  ```

- To convert a `.py` file back to a `.ipynb` file:
  ```
  jupytext --to ipynb <notebook>.py
  ```

Note that you can prepend `uv run` to these commands if you have `jupytext` installed in your `uv`-managed virtual environment.

## Building Documentation Locally

If you made changes that affect the documentation, you can build the docs locally to verify your changes by running:

```
uv run mkdocs serve
```

This starts a local development server you can view at http://127.0.0.1:8000. If you want to add a new example notebook, make sure to convert it to `.py` format using `jupytext` as described above, and place it in the `examples` directory. When the docs are built, the notebook will be run and converted to `.md` format automatically and placed in an `_examples` directory. In order to link to the notebook in the documentation, make sure to add it to the `Examples` section under `nav` in the `mkdocs.yml` file.
If you don't want to execute the example notebooks everytime the docs are built locally, comment the line `- docs/convert_examples.py` in `mkdocs.yml`.
