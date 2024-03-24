# HuBMAP
[HuBMAP - Hacking the Human Vasculature Segment instances of microvascular structures from healthy human kidney tissue slides.](https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature)

## Goal of the Competition
The goal of this competition is to segment instances of microvascular structures, including capillaries, arterioles, and venules. You'll create a model trained on 2D PAS-stained histology images from healthy human kidney tissue slides.

Your help in automating the segmentation of microvasculature structures will improve researchers' understanding of how the blood vessels are arranged in human tissues.

## Context
The proper functioning of your body's organs and tissues depends on the interaction, spatial organization, and specialization of your cells—all 37 trillion of them. With so many cells, determining their functions and relationships is a monumental undertaking.

Current efforts to map cells involve the Vasculature Common Coordinate Framework (VCCF), which uses the blood vasculature in the human body as the primary navigation system. The VCCF crosses all scale levels--from the whole body to the single cell level--and provides a unique way to identify cellular locations using capillary structures as an address. However, the gaps in what researchers know about microvasculature lead to gaps in the VCCF. If we could automatically segment microvasculature arrangements, researchers could use the real-world tissue data to begin to fill in those gaps and map out the vasculature.

Competition host Human BioMolecular Atlas Program (HuBMAP) hopes to develop an open and global platform to map healthy cells in the human body. Using the latest molecular and cellular biology technologies, HuBMAP researchers are studying the connections that cells have with each other throughout the body.

There are still many unknowns regarding microvasculature, but your Machine Learning insights could enable researchers to use the available tissue data to augment their understanding of how these small vessels are arranged throughout the body. Ultimately, you'll be helping to pave the way towards building a Vascular Common Coordinate Framework (VCCF) and a Human Reference Atlas (HRA), which will identify how the relationships between cells can affect our health.

## Python Project

For details see the [`pyproject.toml`](./pyproject.toml) file.
We use [`poetry`](https://python-poetry.org/) as the python package manager.

### Activate Virtual Environment

If you are not using `conda` to manage virtual environments, `cd` into the project root [`./`](./) and run the following command.

```bash
$ poetry shell
```

### Add Dependency

```bash
$ poetry add <PACKAGE_NAME>
```

### Install all Dependencies

```bash
$ poetry install
```

### Export as `requirements.txt`

```bash
$ poetry export --without-hashes --format=requirements.txt > requirements.txt
```
