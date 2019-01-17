import tensorflow as tf


def init_sess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    return sess


def text_to_code(tokens, dict, seq_len):
    code_str = ""
    eof_code = len(dict)
    for sentence in tokens:
        index = 0
        for word in sentence:
            try:
                id = dict[word]
            except KeyError:
                id = eof_code
            code_str += (str(id) + ' ')
            index += 1
        while index < seq_len:
            code_str += (str(eof_code) + ' ')
            index += 1
        code_str += '\n'
    return code_str


def code_to_text(codes, dict):
    paras = ""
    eof_code = len(dict)
    for line in codes:
        numbers = map(int, line)
        for number in numbers:
            if number == eof_code:
                continue
            paras += (dict[str(number)] + ' ')
        paras += '\n'
    return paras


def get_tokens(file):
    tokenized = list()
    with open(file, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tokenized.append(line.strip().split())
    return tokenized


def get_word_list(tokens):
    word_set = list()
    for sentence in tokens:
        for word in sentence:
            word_set.append(word)
    return list(set(word_set))


def get_dict(word_set):
    word_index_dict = dict()
    index_word_dict = dict()
    index = 0
    for word in word_set:
        word_index_dict[word] = str(index)
        index_word_dict[str(index)] = word
        index += 1
    return word_index_dict, index_word_dict


def text_precess(train_text, test_text=None):
    train_tokens = get_tokens(train_text)
    if test_text is None:
        test_tokens = list()
    else:
        test_tokens = get_tokens(test_text)
    word_set = get_word_list(train_tokens + test_tokens)
    word_index_dict, index_word_dict = get_dict(word_set)

    if test_text is None:
        sequence_len = len(max(train_tokens, key=len))
    else:
        sequence_len = max(len(max(train_tokens, key=len)), len(max(test_tokens, key=len)))
    # with open('save/eval_data.txt', 'w') as outfile:
    #     outfile.write(text_to_code(test_tokens, word_index_dict, sequence_len))

    return sequence_len, len(word_index_dict) + 1, word_index_dict, index_word_dict

