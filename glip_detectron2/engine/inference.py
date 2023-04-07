import re
import yaml
import logging
from collections import defaultdict


def create_queries_and_maps_from_dataset(dataset, cfg, tokenizer=None):
    # reimplemented this method to allow supplying a custom tokenizer
    categories = dataset.names
    # one_hot = dataset.one_hot

    labels = []
    label_list = []
    keys = list(categories.keys())
    keys.sort()
    for i in keys:
        labels.append(i)
        label_list.append(categories[i])

    if cfg.TEST.CHUNKED_EVALUATION != -1:
        labels = chunks(labels, cfg.TEST.CHUNKED_EVALUATION)
        label_list = chunks(label_list, cfg.TEST.CHUNKED_EVALUATION)
    else:
        labels = [labels]
        label_list = [label_list]

    all_queries = []
    all_positive_map_label_to_token = []

    for i in range(len(labels)):
        labels_i = labels[i]
        label_list_i = label_list[i]
        query_i, positive_map_label_to_token_i = create_queries_and_maps(
            labels_i, label_list_i,
            additional_labels=cfg.DATASETS.SUPRESS_QUERY if cfg.DATASETS.USE_SUPRESS_QUERY else None, cfg=cfg,
            tokenizer=tokenizer)

        all_queries.append(query_i)
        all_positive_map_label_to_token.append(positive_map_label_to_token_i)
    logging.info("All queries: " + str(all_queries))
    return all_queries, all_positive_map_label_to_token


def create_queries_and_maps(labels, label_list, additional_labels=None, cfg=None, tokenizer=None):
    # reimplemented this method to allow supplying a custom tokenizer
    # Clean label list
    label_list = [clean_name(i) for i in label_list]
    # Form the query and get the mapping
    tokens_positive = []
    objects_query = ""

    # sep between tokens, follow training
    token_separation = cfg.DATASETS.TOKEN_SEPARATION if cfg is not None else ". "

    prompt_template = None
    if cfg is not None and cfg.DATASETS.OVERRIDE_CLASS_NAMES \
            and 'prompt_template' in cfg.DATASETS.OVERRIDE_CLASS_NAMES[0]:
        prompt_template = []
        for cat in sorted(cfg.DATASETS.OVERRIDE_CLASS_NAMES, key=lambda x: x['id']):
            prompt_template.append(cat.get('prompt_template_val', cat['prompt_template']))

    # if prompt_template is not None and isinstance(prompt_template, str):
    #     prompt_template = load_from_yaml_file(prompt_template)
    use_prompt_template = prompt_template is not None and cfg.DATASETS.USE_PROMPT_TEMPLATE
    for _index, label in enumerate(label_list):
        if use_prompt_template:
            objects_query += prompt_template[_index]
            start_i = objects_query.index('{name}')
            objects_query = objects_query.format(name=label)
        else:
            start_i = len(objects_query)
            objects_query += label

        end_i = start_i + len(label)
        tokens_positive.append([(start_i, end_i)])  # Every label has a [(start, end)]

        if _index != len(label_list) - 1:
            objects_query += token_separation

    if additional_labels is not None:
        objects_query += token_separation
        for _index, label in enumerate(additional_labels):
            objects_query += label
            if _index != len(additional_labels) - 1:
                objects_query += token_separation

    if tokenizer is None:
        from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "bert-base-uncased":
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            tokenized = tokenizer(objects_query, return_tensors="pt")
        elif cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
            from transformers import CLIPTokenizerFast
            if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
                tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                              from_slow=True, mask_token='ðŁĴĳ</w>')
            else:
                tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                              from_slow=True)
            tokenized = tokenizer(objects_query,
                                  max_length=cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                                  truncation=True,
                                  return_tensors="pt")
        else:
            raise NotImplementedError
    else:
        if cfg is None or cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "bert-base-uncased":
            tokenized = tokenizer(objects_query, return_tensors="pt")
        elif cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
            tokenized = tokenizer(objects_query,
                                  max_length=cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                                  truncation=True,
                                  return_tensors="pt")
        else:
            raise NotImplementedError

    # Create the mapping between tokenized sentence and the original label
    positive_map_token_to_label, positive_map_label_to_token = create_positive_dict(tokenized, tokens_positive,
                                                                                    labels=labels)  # from token position to original label
    return objects_query, positive_map_label_to_token


def create_positive_dict(tokenized, tokens_positive, labels):
    """construct a dictionary such that positive_map[i] = j, iff token i is mapped to j label"""
    positive_map = defaultdict(int)

    # Additionally, have positive_map_label_to_tokens
    positive_map_label_to_token = defaultdict(list)
    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            for i in range(beg_pos, end_pos + 1):
                positive_map[i] = labels[j]
                positive_map_label_to_token[labels[j]].append(i)
    return positive_map, positive_map_label_to_token


def load_from_yaml_file(yaml_file):
    with open(yaml_file, 'r') as fp:
        return yaml.load(fp, Loader=yaml.CLoader)


def clean_name(name):
    name = re.sub(r"\(.*\)", "", name)
    # name = re.sub(r"_", " ", name)
    name = re.sub(r" {2}", " ", name)
    return name


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert(counter == len(lst))

    return all_
