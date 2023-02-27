import os
import re
import glob
from typing import List
import neologdn
import unicodedata
import demoji
import MeCab
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# 責務 : ファイルのパスを引数に受け取り、ファイルの中身を文字列として返す
def get_text_from_file(file_path: str) -> str:
  with open(file_path, 'r') as f:
    return f.read()

def clean_text(text: str) -> str:
  # 改行コード除去
  text = text.replace('\n', '').replace('\r', '')

  # URL除去
  text = re.sub(r'http?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
  text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)

  # 絵文字除去
  text = demoji.replace(string=text, repl='')

  # 半角記号除去
  text = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]').sub('', text)

  # 全角記号除去
  text = re.sub('[\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F]', '', text)

  # 全角空白の除去
  text = re.sub(r'　', ' ', text)

  return text

def normarize_text(text: str) -> str:
  # 長音記号除去
  text = neologdn.normalize(text)

  # 英数字は半角、カタカナは全角に変換
  text = unicodedata.normalize('NFKC', text)

  # 小文字に統一
  text = text.lower()

  # 桁区切り数字を 0 に変換
  text = re.sub(r'\b\d{1,3}(,\d{3})*\b', '0', text)

  # 数値を全て 0 に変換
  text = re.sub(r'\d+', '0', text)

  return text

def pretreatment(text: str) -> str:
  text = clean_text(text)
  text = normarize_text(text)

  return text

def morphological_analysis(text: str) -> List[str]:
  tagger = MeCab.Tagger('-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
  node = tagger.parseToNode(text)
  
  words: List[str] = []
  while node:

    if (node.surface != ''):
      if node.feature.split(',')[6] == '*':
        words.append(node.surface)
      elif (node.feature.split(',')[0] != '助詞' and node.feature.split(',')[0] != '助動詞'):
        words.append(node.feature.split(',')[6])
    node = node.next

  return words

def delete_stop_word(words: List[str]) -> List[str]:
  # ひらがな一文字を除去
  words = [w for w in words if re.compile('[\u3041-\u309F]').fullmatch(w) == None]
  
  # `delete_words`に含まれる単語を除去
  delete_words = ['sports', 'watch']
  words = [w for w in words if not(w in delete_words)]

  return words

if __name__ == '__main__':
  current_path = os.path.dirname(__file__)

  corpus_texts_path = os.path.join(current_path, '../sports-watch/*.txt')
  corpus_file_paths = glob.glob(corpus_texts_path)
  corpus_file_names = list(map(lambda file_name: file_name.split('/')[len(file_name.split('/')) - 1], corpus_file_paths))

  words_list: List[List[str]] = []
  for file_path in corpus_file_paths:
    text = get_text_from_file(file_path)
    replaced_text = pretreatment(text)
    words = morphological_analysis(replaced_text)
    words = delete_stop_word(words)
    words_list.append(words)

  documents = [TaggedDocument(doc, [corpus_file_names[i]]) for i, doc in enumerate(words_list)]
  model = Doc2Vec(documents=documents, dm=0, vector_size=100, window=5)
  
  model.save(os.path.join(current_path, '../models/doc2Vec.model'))

  # model = Doc2Vec.load(os.path.join(current_path, '../models/doc2vec.model'))
  print(model.dv.most_similar('sports-watch-4627799.txt'))