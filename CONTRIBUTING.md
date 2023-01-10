# Contributing

Thanks for taking the time to contribute!

From opening a bug report to creating a pull request: every contribution is
appreciated and welcome. If you're planning to implement a new feature or change
the API, please create an issue first. This way we can ensure that your precious
work is not in vain.


## Setup the development environment

- Clone the repo and go to your local `deel-lip` folder
- Create a virtual environment and install the development requirements:

  `make prepare-dev && source deel_lip_dev_env/bin/activate`.
- You are ready to install the library:

  `pip install -e .`

Welcome to the team!


## Check your code changes

Before opening a pull request, please make sure you check your code and you run the
unit tests:

```bash
$ make test
```

This command will:
- check your code with black PEP-8 formatter and flake8 linter.
- run `unittest` on the `tests/` folder with different Python and TensorFlow versions.


## Submitting your changes

You can submit a pull request. Something that will increase the chance that your pull
request is accepted:

- Write tests and ensure that the existing ones pass.
- If `make test` is successful, you have fair chances to pass the CI workflows (linting
  and tests)
- Write a [good commit message](https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html) (we follow a lowercase convention).
- For a major fix/feature make sure your PR has an issue and if it doesn't, please
  create one. This would help discussion with the community, and polishing ideas in case
  of a new feature.
