import re
import jieba
from gensim.models import word2vec
from TextCNN_code_multi.data_utils import seg_words, create_dict, get_labal_weight,\
    shuffle_padding, sentence_word_to_index, get_vector_tfidf, BatchManager, get_max_len,\
    get_weights_for_current_batch, compute_confuse_matrix
from TextCNN_code_multi.utils import load_data_from_csv, get_tfidf_and_save, load_tfidf_dict,\
    load_word_embedding


train_data_path = "../data/sentiment_analysis_trainingset.csv"
dev_data_path = "../data/sentiment_analysis_validationset.csv"
path_word2vec_word_string = "data/word2vec_word_string.txt"   # 用于训练word的word2vec的语料库


def get_word_data():
    train_data_df = load_data_from_csv(train_data_path)
    validate_data_df = load_data_from_csv(dev_data_path)
    content_train = train_data_df.iloc[:, 1]
    content_valid = validate_data_df.iloc[:, 1]
    string_train = seg_words(content_train, "word")
    string_train = " ".join(string_train)
    string_valid = seg_words(content_valid, "word")
    string_valid = " ".join(string_valid)
    string = string_train + " " + string_valid
    with open(path_word2vec_word_string, "w", encoding="utf-8") as f:
        f.write(string)


def get_word2vec(type):
    if type == "word":
        sentences = word2vec.LineSentence(path_word2vec_word_string)
        model = word2vec.Word2Vec(sentences, sg=0, hs=1, min_count=1, window=4, size=100, iter=20)  # CBOW
        model.save("data/word2vec_word_model")
        model.wv.save_word2vec_format('data/word2vec_word_model.txt', binary=False)
    else:
        pass
        # sentences = word2vec.LineSentence(path_word2vec_char_string)
        # model = word2vec.Word2Vec(sentences, sg=0, hs=1, min_count=1, window=4, size=100, iter=20)  # CBOW
        # model.save("data/word2vec_char_model")
        # model.wv.save_word2vec_format('data/word2vec_char_model.txt', binary=False)


if __name__ == "__main__":
    get_word_data()
    get_word2vec("word")
