import pandas as pd
import numpy as np
import re
import string
import copy
import streamlit as st
import spacy
import ginza
import sudachipy
from sudachipy import tokenizer   
from sudachipy import dictionary  

#　類似度の取得
def result():
    search = st.session_state.search
    search = search.replace(" ","")
    search = search.replace("　","")
    st.write('検索結果')
    if search == '':
        #　何も入力されていない場合に表示
        st.write("結果なし")
    
    else:
        nlp = spacy.load("ja_ginza")

        tokenizer_obj = dictionary.Dictionary().create()
        mode = tokenizer.Tokenizer.SplitMode.B

        splitter = nlp.get_pipe("compound_splitter")
        splitter.split_mode = 'C'

        data_list = copy.deepcopy(st.session_state.dl)
        text_list = copy.deepcopy(st.session_state.tl)
        tag_list = copy.deepcopy(st.session_state.tag)
        word_vec_list = copy.copy(st.session_state.wv)

        normalized_words = [m.normalized_form() for m in tokenizer_obj.tokenize(search, mode)]
        normalized_sentence = ' '.join(normalized_words)
        word_vec_kw = nlp(normalized_sentence)

        # for sentence in word_vec_kw.sents:
        #     for token in sentence:
        #         st.write(token.text)
        #         st.write(token.vector[0:2])

        # 文章同士のコサイン類似度を求める
        sim_list = []
        for i in range(len(data_list)):
            sim = word_vec_kw.similarity(word_vec_list[i])
            sim_list.append(sim)

        sim_list = list(map(float, sim_list))
        result = list(zip(sim_list, text_list, tag_list))

        tag = []
        for i in range(len(result)):    
            _, _, x = result[i]
            tag.append(x)

        req_list = copy.deepcopy(st.session_state.req)
        if len(req_list) >= 1:
            left_list = []
            for i in range(len(req_list)):
                for n in range(len(result)):
                    if req_list[i] in tag[n]:
                        left_list.append(result[n])
            
            sort_result = sorted(left_list, reverse=True)
        
        if len(req_list) < 1:
            sort_result = sorted(result, reverse=True)

        sort_result = list(dict.fromkeys(sort_result))

        for sim_list, text_list, tag_list in sort_result:
          st.write('類似度: ' + str(sim_list))
          st.write('文章: ' + text_list)

def main():
    st.title("検索システム（仮)")

    uploaded_file = st.sidebar.file_uploader("データのアップロード", type='csv')


    if uploaded_file is not None:

        ufile = copy.deepcopy(uploaded_file)

        try:
            # 文字列の判定
            pd.read_csv(ufile, encoding="utf_8_sig")
            enc = "utf_8_sig"
        except:
            enc = "shift-jis"

        finally:
            # データフレームの読み込み
            df = pd.read_csv(uploaded_file, encoding=enc) 
            pd.set_option("display.max_colwidth", 500)

            df_data = df

            # 比較用データリスト
            data_list = []
            for index, data in df_data.iterrows():
                data_list.append(data['質問事項'] + data['回答'])
            
            # データリストの不要な文字を置き換え
            for i in range(len(data_list)):
                data_list[i] = data_list[i].replace('\r','')
                data_list[i] = data_list[i].replace('\n','')
                data_list[i] = data_list[i].translate(str.maketrans('','', string.punctuation))
                data_list[i] = data_list[i].replace('、','')
                data_list[i] = data_list[i].replace('・','')
                data_list[i] = data_list[i].replace('。','')

            # 表示用テキストリスト
            text_list = []
            for index, data in df_data.iterrows():
                text_list.append(data['回答'])

            #タグ取得用テキストリスト
            tag_list = []
            for index, data in df_data.iterrows():
                tag_list.append(data['タグ'])

            nlp = spacy.load("ja_ginza")
            
            tokenizer_obj = dictionary.Dictionary().create()
            mode = tokenizer.Tokenizer.SplitMode.B

            splitter = nlp.get_pipe("compound_splitter")
            splitter.split_mode = 'C'

            word_vec_list =  []
            for i in range(len(data_list)):
                normalized_words = [m.normalized_form() for m in tokenizer_obj.tokenize(data_list[i], mode)]
                normalized_sentence = ' '.join(normalized_words)
                doc_A = nlp(normalized_sentence)
                word_vec_list.append(doc_A)

            # データフレームをセッションステートに退避
            st.session_state.df = copy.deepcopy(df)
            st.session_state.dl = copy.deepcopy(data_list)
            st.session_state.tl = copy.deepcopy(text_list)
            st.session_state.tag = copy.deepcopy(tag_list)
            st.session_state.wv = copy.copy(word_vec_list)

            req_list = []

            s1 = st.sidebar.checkbox('マーケティング')
            if s1:
                req_list.append("マーケティング")
            s2 = st.sidebar.checkbox('ビジネスプラン')
            if s2:
                req_list.append("ビジネスプラン")
            s3 = st.sidebar.checkbox('資金計画')
            if s3:
                req_list.append("資金計画")

            st.session_state.req = copy.deepcopy(req_list)

            st.text_input("検索入力欄", key="search")
            st.caption("入力例（資金調達　方法）（起業　資金）")
            if st.button("検索"):
                result()

    else:
        st.subheader('データをアップロードしてください')
        
                

if __name__ == "__main__":
    main()
