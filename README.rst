
pybiber: Aggregate counts of linguistic features retrieved from spaCy parsing based on Biber's taxonomy
=======================================================================================================
|pypi| |pypi_downloads| |tests|

The pybiber package aggregates the lexicogrammatical and functional features described by `Biber (1988) <https://books.google.com/books?id=CVTPaSSYEroC&dq=variation+across+speech+and+writing&lr=&source=gbs_navlinks_s>`_ and widely used for text-type, register, and genre classification tasks.

The package uses `spaCy <https://spacy.io/models>`_ part-of-speech tagging and dependency parsing to summarize and aggregate patterns.

Because feature extraction builds from the outputs of probabilistic taggers, the accuracy of the resulting counts are reliant on the accuracy of those models. Thus, texts with irregular spellings, non-normative punctuation, etc. will likely produce unreliable outputs, unless taggers are tuned specifically for those purposes.

See `the documentation <https://browndw.github.io/pybiber>`_ for description of the package's full functionality.

See `pseudobibeR <https://cran.r-project.org/web/packages/pseudobibeR/index.html>`_ for the R implementation.

Installation
------------

You can install the released version of pybiber from `PyPI <https://pypi.org/project/pybiber/>`_:

.. code-block:: install-pybiber

    pip install pybiber

Install a `spaCY model <https://spacy.io/usage/models#download>`_:

.. code-block:: install-model

    python -m spacy download en_core_web_sm

Usage
-----

To use the pybiber package, you must first import `spaCy <https://spacy.io/models>`_ and initiate an instance. You will also need to create a corpus. The :code:`biber` function expects a `polars DataFrame <https://docs.pola.rs/api/python/stable/reference/dataframe/index.html>`_ with a :code:`doc_id` column and a :code:`text` column. This follows the convention for `readtext <https://readtext.quanteda.io/articles/readtext_vignette.html>`_ and corpus processing using `quanteda <https://quanteda.io/>`_ in R.

.. code-block:: import

    import spacy
    import pybiber as pb
    from pybiber.data import micusp_mini

The pybiber package requires a model that will carry out part-of-speech tagging and `dependency parsing <https://spacy.io/usage/linguistic-features>`_.

.. code-block:: import

    nlp = spacy.load("en_core_web_sm", disable=["ner"])

To process the corpus, use :code:`spacy_parse`. Processing the :code:`micusp_mini` corpus should take between 20-30 seconds.

.. code-block:: import

    df_spacy = pb.spacy_parse(micusp_mini, nlp)

After parsing the corpus, features can then be aggregated using :code:`biber`.

.. code-block:: import

    df_biber = pb.biber(df_spacy)

Pipeline (one-liner)
--------------------

If you want to go straight from a folder of .txt files to Biber features, use the high-level pipeline:

.. code-block:: import

    from pybiber import PybiberPipeline

    pipeline = PybiberPipeline(model="en_core_web_sm", disable_ner=True)
    # Read, parse, and compute features; set return_tokens=True to get token table too
    df_biber = pipeline.run_from_folder("/path/to/texts")

You can also run the pipeline on an in-memory corpus:

.. code-block:: import

    df_biber, df_tokens = pipeline.run(corpus_df, return_tokens=True)

License
-------

Code licensed under the `MIT License <https://opensource.org/license/mit/>`_.
See the `LICENSE <https://github.com/browndw/pybiber/blob/master/LICENSE>`_ file.

.. |pypi| image:: https://badge.fury.io/py/pybiber.svg
    :target: https://badge.fury.io/py/pybiber
    :alt: PyPI Version

.. |pypi_downloads| image:: https://img.shields.io/pypi/dm/pybiber
    :target: https://pypi.org/project/pybiber/
    :alt: Downloads from PyPI

.. |tests| image:: https://github.com/browndw/pybiber/actions/workflows/test.yml/badge.svg
    :target: https://github.com/browndw/pybiber/actions/workflows/test.yml
    :alt: Test Status
