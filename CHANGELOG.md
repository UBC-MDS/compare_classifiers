# Changelog

All notable changes to this project will be documented in this file. See [standard-version](https://github.com/conventional-changelog/standard-version) for commit guidelines.

### 1.0.1 (2025-02-01)

### Bug Fixes

* addressed by peer review in issue [#78](https://github.com/UBC-MDS/compare_classifiers/issues/78) and [#76](https://github.com/UBC-MDS/compare_classifiers/issues/76): re-arranged order of authors to be alphabetical; reformatted author email addresses in README ([789ed85](https://github.com/UBC-MDS/compare_classifiers/commit/789ed8529612cb0ddae52b326e922bb251088b38))
* addressed by peer review in issue 81 - (continuing from previous fix on the same issue): added link for repo status, python versions and ci/cd badges and text hyperlink to tutorial/documentation in README ([a501a36](https://github.com/UBC-MDS/compare_classifiers/commit/a501a36236074c3a97f1af7f81e95d98a2ce2880))
* feedback addressed by instructor for previous milestones in issue [#74](https://github.com/UBC-MDS/compare_classifiers/issues/74): added order as input param in error checking helper function docstrings ([1bfddf2](https://github.com/UBC-MDS/compare_classifiers/commit/1bfddf232153ffbb40c54fed5aae4f443568decd))
* feedback addressed by instructor in previous milestones in issue [#75](https://github.com/UBC-MDS/compare_classifiers/issues/75): removed fig from confusion_matrices returned objects and updated docstring; had pytest not display plots when testing; updated example.ipynb to not show function returned object and fixed error in first code block ([18ce87d](https://github.com/UBC-MDS/compare_classifiers/commit/18ce87dda827e3c69937d0278410ed8c941da09a))
* Feedback addressed by nvarabioff ([ceefbdc](https://github.com/UBC-MDS/compare_classifiers/commit/ceefbdc66b2ccc178d1d2a16e2db086a7a02487a))
* Feedback addressed by nvarabioff ([ff3ee4c](https://github.com/UBC-MDS/compare_classifiers/commit/ff3ee4cf9ee50f6d6823de9b2be84aa68b24683a))
* Feedback addressed by nvarabioff ([08e1105](https://github.com/UBC-MDS/compare_classifiers/commit/08e11051adb5d31a288aa46aa23087b1de74282b))
* Feedback addressed by nvarabioff ([ab269bb](https://github.com/UBC-MDS/compare_classifiers/commit/ab269bbbe655aca139e0ed4a33bfdacedcf7d423))
* feedback addressed by peer review in issue [#84](https://github.com/UBC-MDS/compare_classifiers/issues/84): added contributor email addresses in README ([32c2bb3](https://github.com/UBC-MDS/compare_classifiers/commit/32c2bb3484f021fc3f9e2d195db2f4fa5331acc4))
* feedback addressed by peer views in issue [#87](https://github.com/UBC-MDS/compare_classifiers/issues/87): made code in example.ipynb and README runable; updated author name orders in pyproject.toml to match rest of the project; uploaded example dataset as .csv to root dir and added in explanation of dataset in README ([9c9f608](https://github.com/UBC-MDS/compare_classifiers/commit/9c9f6080286977036a80a4cc82f040df8f08d4a6))


### Issues Addressed without Explicit Commits

* issue [#83](https://github.com/UBC-MDS/compare_classifiers/issues/83): package version has been added to CHANGELOG automatically as it is updated with continuous deploymnent.
* issue issue [#79](https://github.com/UBC-MDS/compare_classifiers/issues/79): package has been published to PyPI and succesfully installation has now been verified.

### Unaddressed Issues

Besides above fixes, we have already resolved some issues mentioned in previous instructor milestone feedback during Milestone 2 and 3. Below are some low-priority issues we decided to not address at the moment, as seen in issue [#77](https://github.com/UBC-MDS/compare_classifiers/issues/77).

* Error handling functions, such as `check_valid_X.py`, are designed to accept all types of input for `X` in order to throw exceptions for unsupported input types, therefore docstring indicating `X` as Any would be correct, and no fix is needed. Same goes for `check_valid_y.py` and `check_valid_estimators.py`.
* Warnings caused by models during tests are not handled at the moment, since it is part of the developer functionalities and not user facing. However, these issues should be handled in a future version when time is permitted.
* Though there is slight overlap between `compare_f1.py` and `ensemble_compare_f1.py`, it is not enough be refactored into a new helper function without generating significantly more lines of code, therefore no further change should be made.
* Using the same set of test data across all test functions is a good idea, though we did think using more than one dataset improves robustness of testing. We need to think of a solution in the future to include a variety of test data so we don't only rely on one single dataset for testing. For now, we will leave this as is because we have not come up with a solution.

## v0.1.0 (07/01/2025)

- First release of `compare_classifiers`!