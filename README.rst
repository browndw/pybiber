
pybiber: Aggregate counts of linguistic features retrieved from spaCy parsing based on Biber's taxonomy
=======================================================================================================
|pypi| |pypi_downloads|

The pybiber package aggregates the lexicogrammatical and functional features described by `Biber (1991) <https://books.google.com/books?id=CVTPaSSYEroC&dq=variation+across+speech+and+writing&lr=&source=gbs_navlinks_s>`_ and widely used for text-type, register, and genre classification tasks.

The package uses `spaCy <https://spacy.io/models>`_ part-of-speech tagging and dependency parsing to summarize and aggregate patterns.

Because feature extraction builds from the outputs of probabilistic taggers, the accuracy of the resulting counts are reliant on the accuracy of those models. Thus, texts with irregular spellings, non-normative punctuation, etc. will likely produce unreliable outputs, unless taggers are tuned specifically for those purposes.

Installation
------------

The package can be installed with the :code:`en_core_web_sm` base model:

.. code-block:: with-model

    pip install pybiber[model]

Or without a model:

.. code-block:: without-model

    pip install pybiber

Usage
-----

To use the pybiber package, you must first import `spaCy <https://spacy.io/models>`_ and initiate an instance. You will also need to create a corpus. The :code:`biber` function expects a `polars DataFrame <https://docs.pola.rs/api/python/stable/reference/dataframe/index.html>`_ with a :code:`doc_id` column and a :code:`text` column. This follows the convention for `readtext <https://readtext.quanteda.io/articles/readtext_vignette.html>`_ and corpus processing using `quanteda <https://quanteda.io/>`_ in R.


License
-------

Code licensed under `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.
See `LICENSE <https://github.com/browndw/docuscospacy/blob/master/LICENSE>`_ file.

.. |pypi| image:: https://badge.fury.io/py/pybiber.svg
    :target: https://badge.fury.io/py/pybiber
    :alt: PyPI Version

.. |pypi_downloads| image:: https://img.shields.io/pypi/dm/pybiber
    :target: https://pypi.org/project/pybiber/
    :alt: Downloads from PyPI

