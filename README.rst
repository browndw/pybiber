
pybiber: A comprehensive Python package for linguistic feature extraction and Multi-Dimensional Analysis
========================================================================================================
|pypi| |pypi_downloads| |tests|

The pybiber package provides tools for extracting 67 lexicogrammatical and functional features described by `Biber (1988) <https://books.google.com/books?id=CVTPaSSYEroC&dq=variation+across+speech+and+writing&lr=&source=gbs_navlinks_s>`_ and widely used for text-type, register, and genre classification tasks in corpus linguistics.

**Key Features:**

- **67 Linguistic Features**: Automated extraction of tense markers, pronouns, subordination patterns, modal verbs, and more
- **Multi-Dimensional Analysis**: Complete implementation of Biber's MDA methodology for register analysis
- **Principal Component Analysis**: Alternative dimensionality reduction approaches with visualization tools
- **High Performance**: Built on `spaCy <https://spacy.io/models>`_ and `Polars <https://docs.pola.rs/>`_ for efficient text processing
- **End-to-End Pipeline**: From raw text files to statistical analysis in just a few lines of code
- **Comprehensive Visualization**: Built-in plotting functions for exploratory data analysis

**Applications:**

- Register and genre analysis in corpus linguistics
- Text classification and machine learning preprocessing  
- Diachronic language change studies
- Cross-linguistic variation research
- Academic writing analysis and pedagogical applications
- Stylometric analysis and authorship attribution

The package uses `spaCy <https://spacy.io/models>`_ part-of-speech tagging and dependency parsing with `Polars <https://docs.pola.rs/>`_ DataFrames for high-performance analytics.

**Accuracy Note**: Feature extraction builds from probabilistic taggers, so accuracy depends on model quality. Texts with irregular spellings or non-standard punctuation may produce unreliable outputs unless taggers are specifically tuned for those domains.

See `the documentation <https://browndw.github.io/pybiber>`_ for comprehensive guides and API reference.

See `pseudobibeR <https://cran.r-project.org/web/packages/pseudobibeR/index.html>`_ for the R implementation.

Quick Start
-----------

**One-line processing** from a folder of text files:

.. code-block:: python

    import pybiber as pb

    # Process all .txt files in a directory
    pipeline = pb.PybiberPipeline(model="en_core_web_sm")
    features = pipeline.run_from_folder("path/to/texts")

**Multi-Dimensional Analysis** with visualization:

.. code-block:: python

    # Create analyzer for statistical analysis
    analyzer = pb.BiberAnalyzer(features)
    
    # Perform MDA and generate scree plot
    mda_results = analyzer.mda()
    analyzer.mdaviz_screeplot()
    
    # Plot group means by dimension
    analyzer.mdaviz_groupmeans(grouping_var="register")

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

**Data Requirements**

The pybiber package works with corpora structured as DataFrames with:
- ``doc_id`` column: Unique document identifiers  
- ``text`` column: Raw text content

This follows conventions from `readtext <https://readtext.quanteda.io/articles/readtext_vignette.html>`_ and `quanteda <https://quanteda.io/>`_.

**Step-by-Step Workflow**

1. **Import libraries and load spaCy model**:

.. code-block:: python

    import spacy
    import pybiber as pb
    from pybiber.data import micusp_mini  # Sample corpus
    
    nlp = spacy.load("en_core_web_sm")

2. **Parse corpus with spaCy**:

.. code-block:: python

    # Parse texts to extract linguistic annotations (modern approach)
    processor = pb.CorpusProcessor()
    tokens_df = processor.process_corpus(micusp_mini, nlp)

3. **Extract Biber features**:

.. code-block:: python

    # Aggregate 67 linguistic features per document  
    features_df = pb.biber(tokens_df)

4. **Advanced Analysis** (optional):

.. code-block:: python

    # Statistical analysis and visualization
    analyzer = pb.BiberAnalyzer(features_df)
    
    # Multi-Dimensional Analysis
    mda_results = analyzer.mda()
    
    # Principal Component Analysis
    pca_results = analyzer.pca()
    
    # Visualization options
    analyzer.mdaviz_screeplot()           # Eigenvalue plot
    analyzer.pcaviz_contrib()             # Feature contributions
    analyzer.mdaviz_groupmeans(group_var="genre")  # Group comparisons

**Pipeline Convenience Functions**

For streamlined processing, use the high-level pipeline:

.. code-block:: python

    from pybiber import PybiberPipeline
    
    pipeline = PybiberPipeline(model="en_core_web_sm", disable_ner=True)
    
    # From folder of .txt files
    features_df = pipeline.run_from_folder("/path/to/texts")
    
    # From in-memory corpus
    features_df, tokens_df = pipeline.run(corpus_df, return_tokens=True)
    
    # One-liner convenience functions
    features_df = pb.run_biber_from_folder("/path/to/texts")
    features_df = pb.run_biber(corpus_df)

Feature Categories
------------------

The package extracts 67 linguistic features across 16 categories:

- **Tense & Aspect**: Past tense, perfect aspect, present tense
- **Adverbials**: Place and time adverbials  
- **Pronouns**: 1st/2nd/3rd person, demonstrative, indefinite pronouns
- **Questions**: Direct wh-questions
- **Nominal Forms**: Nominalizations, gerunds, nouns
- **Passives**: Agentless and by-passives
- **Stative Forms**: *be* as main verb, existential *there*
- **Subordination**: 18 different clause types (that-clauses, wh-clauses, infinitives, relatives, etc.)
- **Modification**: Prepositional phrases, attributive/predicative adjectives, adverbs
- **Lexical Specificity**: Type-token ratio, word length
- **Lexical Classes**: Conjuncts, hedges, amplifiers, emphatics, discourse particles
- **Modals**: Possibility, necessity, and predictive modals
- **Specialized Verbs**: Public, private, suasive verbs
- **Reduced Forms**: Contractions, deletions, split constructions
- **Coordination**: Phrasal and clausal coordination
- **Negation**: Synthetic and analytic negation

See the `full feature list <https://browndw.github.io/pybiber/feature-categories.html>`_ for detailed descriptions.

Performance & Requirements
--------------------------

**System Requirements:**
- Python 3.10+
- spaCy model with POS tagging and dependency parsing (e.g., ``en_core_web_sm``)

**Performance Notes:**
- Built on Polars for fast DataFrame operations
- Supports multiprocessing for large corpora
- Memory-efficient processing with configurable batch sizes
- Processing time: ~20-30 seconds for small corpora (e.g., 500 documents)

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
