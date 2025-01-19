
pybiber: Aggregate counts of linguistic features retrieved from spaCy parsing based on Biber's taxonomy
=======================================================================================================

The pybiber package aggregates the lexicogrammatical and functional features described by `Biber (1991) <https://books.google.com/books?id=CVTPaSSYEroC&dq=variation+across+speech+and+writing&lr=&source=gbs_navlinks_s>`_ and widely used for text-type, register, and genre classification tasks.

The package uses `spaCy <https://spacy.io/models>`_ part-of-speech tagging and dependency parsing to summarize and aggregate patterns.

Because feature extraction builds from the outputs of probabilistic taggers, the accuracy of the resulting counts are reliant on the accuracy of those models. Thus, texts with irregular spellings, non-normative punctuation, etc. will likely produce unreliable outputs, unless taggers are tuned specifically for those purposes.

Installation
------------

The package can be installed with the :bash:`en_core_web_sm` base model:

.. code-block:: with-model

    pip install pybiber[model]

Or without a model:

.. code-block:: without-model

    pip install pybiber

License
-------

Code licensed under `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.
See `LICENSE <https://github.com/browndw/docuscospacy/blob/master/LICENSE>`_ file.

.. role:: bash(code)
   :language: bash
