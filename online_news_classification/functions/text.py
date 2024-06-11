import re

import nltk


def truecase(text, only_proper_nouns=False):
    # Apply POS-tagging.
    tagged_sent = nltk.pos_tag([word.lower() for word in nltk.word_tokenize(text)])
    # Infer capitalization from POS-tags.
    capitalize_tags = {"NNP", "NNPS"} if only_proper_nouns else {"NN", "NNS"}
    normalized_sent = [
        word.capitalize() if tag in capitalize_tags else word
        for (word, tag) in tagged_sent
    ]
    # Capitalize first word in sentence.
    normalized_sent[0] = normalized_sent[0].capitalize()
    # Use regular expression to get punctuation right.
    pretty_string = re.sub(" (?=[\\.,'!?:;])", "", " ".join(normalized_sent))
    return pretty_string


def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix) :]
