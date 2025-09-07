"""
Sample text data for testing Biber features.

These samples are designed to test specific linguistic features
that the Biber analyzer should detect.
"""

# Sample texts with known linguistic features
SAMPLE_TEXTS = {
    "quickbrown": "The quick brown fox jumps over the lazy dog.",
    "adj_pred": "The horse is big.",
    "initial_demonstrative": "That is an example sentence.",
    "subordinator_that": "I think he went.",
    "stranded_preposition": "He's the candidate that I was thinking of.",
    "perfect_aspect": "I have written this sentence.",
    "that_verb_comp": "I said that he went.",
    "that_adj_comp": "I'm glad that you like it.",
    "present_participle": ("Stuffing his mouth with cookies, "
                           "Joe ran out the door."),
    "past_participle": ("Built in a single week, the house would "
                        "stand for fifty years."),
    "sentence_relatives": ("Bob likes fried mangoes, which is the most "
                           "disgusting thing I've ever heard of."),
    "wh_question": "When are you leaving?",
    "agentless_passive_1": "The task was done.",
    "agentless_passive_2": ("By beginning this sentence with 'by', "
                            "I may break the classifier."),
    "existential_there": "There is a feature in this sentence.",
    "past_tense": "The cat walked slowly down the street.",
    "place_adverbials": "The meeting is upstairs and outside.",
    "time_adverbials": "I will see you tomorrow and yesterday was great.",
    "modal_verbs": "You should go and might need help.",
    "conditional_subordination": "If you come, we can talk.",
    "causative_subordination": "Because it rained, we stayed inside.",
    "contractions": "I'm sure you won't go there.",
    "nominalizations": ("The establishment of relationships "
                        "requires consideration."),
    "gerunds": "Swimming is fun and running helps.",
    "nouns": "The cat sat on the table.",
    "prepositions": "The book is on the table under the lamp.",
    "attributive_adj": "The big red house stood empty.",
    "predicative_adj": "The house was big and red.",
}

# Expected feature counts for each sample (non-normalized)
# Based on spaCy parsing - these may need adjustment after running actual tests
EXPECTED_FEATURES = {
    "quickbrown": {
        "f_03_present_tense": 1,  # "jumps"
    },
    "adj_pred": {
        "f_41_adj_pred": 1,  # "is big"
    },
    "initial_demonstrative": {
        "f_10_demonstrative_pronoun": 1,  # "That"
    },
    "subordinator_that": {
        "f_60_that_deletion": 1,  # subordinator "that" omitted
    },
    "perfect_aspect": {
        "f_02_perfect_aspect": 1,  # "have written"
    },
    "that_verb_comp": {
        "f_21_that_verb_comp": 1,  # "said that"
    },
    "that_adj_comp": {
        "f_22_that_adj_comp": 1,  # "glad that"
    },
    "present_participle": {
        "f_25_present_participle": 1,  # "Stuffing"
    },
    "sentence_relatives": {
        "f_34_sentence_relatives": 1,  # "which"
    },
    "wh_question": {
        "f_13_wh_question": 1,  # "When"
    },
    "agentless_passive_1": {
        "f_17_agentless_passives": 1,  # "was done"
        "f_18_by_passives": 0,
    },
    "existential_there": {
        "f_20_existential_there": 1,  # "There is"
    },
    "past_tense": {
        "f_01_past_tense": 1,  # "walked"
    },
}
