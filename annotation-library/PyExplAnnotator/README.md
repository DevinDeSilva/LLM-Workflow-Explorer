# PyExplAnnotator

PyExplAnnotator builds RDF/PROV-One knowledge graphs from provenance execution
traces of AI and workflow applications.

The PyPI package name is `pyexplannotator`; the Python import package is
`expl_annotator`.

## Installation

From PyPI, after the package is published:

```bash
pip install pyexplannotator
```

For local development from this directory:

```bash
pip install -e ".[dev]"
```

## Usage

```python
from expl_annotator import ProvOneManager

manager = ProvOneManager("config.yaml")
```

A minimal config file should define the program name and RDF prefixes used by
the graph:

```yaml
program:
  name: my_workflow

ttl:
  save_path: provenance.ttl
  metadata_path: metadata.json
  prefixes:
    - name: my_workflow
      uri: https://example.org/my-workflow/
    - name: prov
      uri: http://www.w3.org/ns/prov#
    - name: provone
      uri: http://purl.org/provone#
    - name: sio
      uri: http://semanticscience.org/resource/
    - name: eo
      uri: https://purl.org/heals/eo#
    - name: xsd
      uri: http://www.w3.org/2001/XMLSchema#
```

## Publishing Checklist

Before publishing, make sure the version in `pyproject.toml` is unique and the
package name `pyexplannotator` is available on PyPI.

Build the source distribution and wheel:

```bash
python -m build
```

Validate the generated distributions:

```bash
python -m twine check dist/*
```

Upload to TestPyPI first:

```bash
python -m twine upload --repository testpypi dist/*
```

Install from TestPyPI in a clean environment:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pyexplannotator
```

When the TestPyPI package looks correct, upload to PyPI:

```bash
python -m twine upload dist/*
```

Use PyPI API tokens for uploads. This repository already ignores `.pypirc`, so
local credentials will not be committed.
