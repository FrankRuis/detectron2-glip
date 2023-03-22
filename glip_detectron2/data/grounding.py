import re
import random
import numpy as np

from pathlib import Path


def get_negatives(path='config/data/lvis_cats_counts.txt'):
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"File {file} does not exist.")

    negatives = []
    for line in file.read_text().splitlines():
        name, count = line.split('\t')
        negatives.append((name, int(count)))
    negs, negs_p = zip(*negatives)
    tot_p = sum(negs_p)
    negs_p = [p / tot_p for p in negs_p]

    return negs, negs_p


def clean_name(name):
    name = re.sub(r"\(.*\)", "", name)
    # name = re.sub(r"_", " ", name)
    name = re.sub(r" {2}", " ", name)
    return name


def convert_od_to_grounding_simple(ind_to_class, shuffle_caption=False, token_separation=" ", caption_prompt=None,
                                   negatives=None, p_negatives=None, sample_negatives=50):
    """
    Convert object detection data into grounding data format, on the fly.
    ind_to_class: {0: "__background__", 1 : "person" ...}, contiguous id
    """
    def generate_sentence_from_labels(positive_label_list, negative_label_list, shuffle=False):
        label_to_positions = {}
        label_list = negative_label_list + positive_label_list
        if shuffle:
            assert (caption_prompt is None), "Should not specify caption_prompt when shuffle is enabled!!"
            random.shuffle(label_list)

        pheso_caption = ""
        for index, label in enumerate(label_list):
            if caption_prompt is not None:
                pheso_caption += caption_prompt[index]['prefix']

            start_index = len(pheso_caption)
            pheso_caption += clean_name(ind_to_class[label] if label in ind_to_class else label)  # NOTE: slight change...
            end_index = len(pheso_caption)

            if caption_prompt is not None:
                pheso_caption += caption_prompt[index]['suffix']

            # e.g.: pheso_caption = "cat dog", where cat is label 4, and dog is label 17
            # label_to_positions: {4: (0, 3), 17: (4, 7)}
            label_to_positions[label if isinstance(label, int) else index] = [start_index, end_index]

            if index != len(label_list) - 1:
                pheso_caption += token_separation

        return label_to_positions, pheso_caption
    if negatives is None:
        negatives, p_negatives = [], []

    label_list = list(sorted(ind_to_class.keys()))  # do not include the background
    if negatives:
        negatives = np.random.choice(negatives, sample_negatives, replace=False, p=p_negatives).tolist()
    label_to_positions, pheso_caption = generate_sentence_from_labels(
        positive_label_list=label_list,
        negative_label_list=negatives,
        shuffle=shuffle_caption
    )
    return label_to_positions, pheso_caption
