import sentencepiece as spm


def train_spm(title_path, model_path,
              vocab_size=2048,
              model_type='unigram',
              character_coverage=0.9995):
    # model_type is in { unigram, bpe, char, word }.
    model_prefix = model_path[:model_path.rfind('.')]
    spm.SentencePieceTrainer.Train(
        f'--input={title_path} '
        f'--model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} '
        f'--model_type={model_type} '
        f'--character_coverage={character_coverage}')


def load_spm(path):
    model = spm.SentencePieceProcessor()
    model.Load(path)
    return model
