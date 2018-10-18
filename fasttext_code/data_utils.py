import re
import jieba


def seg_words(contents, tokenize_style):
    string_segs = []
    if tokenize_style == "word":
        stopwords_set = set()
        for content in contents:
            content = re.sub(" ", "，", content.strip())
            # print(content)
            content = re.sub("\n", "", content.strip())
            segs = jieba.cut(content.strip())
            segs_new = []
            for word in segs:
                if word not in stopwords_set:
                    segs_new.append(word)
                else:
                    pass
            string_segs.append(" ".join(segs_new))
    else:
        for content in contents:
            content = re.sub(" ", "，", content.strip())
            # print(content)
            content = re.sub("\n", "", content.strip())
            # print(" ".join(list(content.strip())))
            string_segs.append(" ".join(list(content.strip())))
    return string_segs
