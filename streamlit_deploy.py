import asyncio
import os
import re
import time
from datetime import datetime, timedelta
from typing import List, Optional, TypedDict

import httpx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter

import pandas as pd
import numpy as np
import streamlit as st
import xmltodict

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents.base import Document
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

def get_recent_months(n=6):
    now = datetime.now()
    return [(now - timedelta(days=30 * i)).strftime("%Y%m") for i in range(n)][::-1]

@st.cache_data
def load_region_codes():
    df = pd.read_csv(
        "국토교통부_법정동코드_20240805.csv",
        encoding="euc-kr",
        dtype={"법정동코드": str},
    )
    df = df[df["폐지여부"] == "존재"]
    return dict(zip(df["법정동명"], df["법정동코드"]))

async def fetch_data_httpx(service_key, lawd_cd, deal_ymd):
    url = "https://apis.data.go.kr/1613000/RTMSDataSvcRHRent/getRTMSDataSvcRHRent"
    params = {"serviceKey": service_key, "LAWD_CD": lawd_cd, "DEAL_YMD": deal_ymd}
    async with httpx.AsyncClient(verify=True, timeout=30.0) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            if not response.text.strip().startswith("<?xml"):
                return pd.DataFrame()
            data_dict = xmltodict.parse(response.content)
            items = (
                data_dict.get("response", {}).get("body", {}).get("items", {}).get("item", [])
            )
            if isinstance(items, dict):
                items = [items]
            df = pd.DataFrame(items)
            df["deal_ymd"] = deal_ymd
            return df
        except Exception as e:
            print(f"{deal_ymd} 월 데이터 조회 실패: {str(e)}")
            return pd.DataFrame()

async def fetch_all_data(service_key, lawd_cd, months):
    tasks = [fetch_data_httpx(service_key, lawd_cd, m) for m in months]
    dfs = await asyncio.gather(*tasks)
    valid_dfs = [df for df in dfs if df is not None and not df.empty]
    if not valid_dfs:
        return pd.DataFrame()
    return pd.concat(valid_dfs, ignore_index=True)

def process_data(df, rent_type):
    if df.empty:
        return df
    if rent_type == "전세":
        df = df[df["monthlyRent"] == "0"]
    else:
        df = df[df["monthlyRent"] != "0"]
    df["보증금"] = df["deposit"].str.replace(",", "").str.strip().astype(int)
    df["월세"] = df["monthlyRent"].str.replace(",", "").str.strip().astype(int)
    df["년월"] = df["deal_ymd"].str[:4] + "-" + df["deal_ymd"].str[4:]
    return df

def plot_trend(df):
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False

    monthly = df.groupby("년월")[["보증금", "월세"]].mean().reset_index()
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    ax.plot(monthly["년월"], monthly["보증금"], marker="o", markersize=10, color="yellow", label="보증금(만원)", linewidth=3, alpha=0.4)
    ax.plot(monthly["년월"], monthly["월세"], marker="s", markersize=10, color="aqua", label="월세(만원)", linewidth=3, alpha=0.4)

    for i in range(len(monthly)):
        ax.annotate(f'{monthly["보증금"].iloc[i]:,.0f}', (monthly["년월"].iloc[i], monthly["보증금"].iloc[i]),
                    textcoords="offset points", xytext=(0,10), ha='center', color="yellow", fontsize=10,fontweight='bold')
        ax.annotate(f'{monthly["월세"].iloc[i]:,.0f}', (monthly["년월"].iloc[i], monthly["월세"].iloc[i]),
                    textcoords="offset points", xytext=(0,10), ha='center', color="aqua", fontsize=10,fontweight='bold')

    ax.set_ylabel("금액(만원)\n", color='white')
    ax.set_xlabel("\n시기", color='white')
    plt.xticks(rotation=90, color='white')
    plt.yticks(color='white')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f'{int(x):,}'))
    ax.legend(facecolor='gray', edgecolor='white', labelcolor='white', loc='upper right')
    ax.grid(True, color='gray', linestyle='--', alpha=0.5)
    return fig

class ChatState(TypedDict):
    input: str
    chat_history: Optional[List[BaseMessage]]
    output: Optional[str]

def make_langgraph_chatbot(df):
    df.fillna("", inplace=True)

    def make_text(row):
        rent_type = "전세" if row["monthlyRent"] == "0" else "월세"
        layer_type = "반지하" if str(row.get("floor", "")) == "-1" else f'{row.get("floor", "")}층'
        return (
            f'{row.get("dealYear", "")}년 {row.get("dealMonth", "")}월 {row.get("dealDay", "")}일, '
            f'{row.get("umdNm", "")} {row.get("jibun", "")}에 위치한 '
            f'{row.get("buildYear", "")}년 준공 "{row.get("mhouseNm", "")}" 건물의 '
            f'{row.get("excluUseAr", "")}㎡ {layer_type}에서 '
            f"{rent_type} 거래가 이루어졌다. "
            f'보증금 {row.get("deposit", "")}만원, 월세 {row.get("monthlyRent", "")}만원이다.'
        )

    docs = [Document(page_content=make_text(row)) for _, row in df.iterrows()]
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vectordb = FAISS.from_documents(chunks, OpenAIEmbeddings())

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "당신은 한국의 전월세 전문가이다. 사용자가 질문하면 전문가로서 사용자에게 적합한 전월세 거래를 추천하라."
        ),
        HumanMessagePromptTemplate.from_template("{context}"),
    ])

    llm = ChatOpenAI(temperature=0)
    qa_chain = create_retrieval_chain(vectordb.as_retriever(), create_stuff_documents_chain(llm, prompt))
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def user_input_node(state: ChatState):
        return {"input": state["input"], "chat_history": state.get("chat_history", [])}

    def response_node(state: ChatState):
        result = qa_chain.invoke(state)
        memory.save_context({"input": state["input"]}, {"output": result["answer"]})
        return {"chat_history": memory.chat_memory.messages, "output": result["answer"]}

    builder = StateGraph(ChatState)
    builder.add_node("user_input", RunnableLambda(user_input_node))
    builder.add_node("generate", RunnableLambda(response_node))
    builder.set_entry_point("user_input")
    builder.add_edge("user_input", "generate")
    builder.add_edge("generate", END)
    return builder.compile()

def main():
    st.set_page_config(layout='wide')
    st.title("🏠 전월세 실거래가 분석 & 대화형 챗봇")

    with st.sidebar:
        st.header("API key 입력")
        openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
        service_encoding_key = st.sidebar.text_input("공공 데이터 API Service Key", type="password")

        if st.button("완료"):
            if not openai_key or not service_encoding_key:
                st.warning("API Key를 모두 입력해주세요.")
            else:
                st.session_state.api_ready = True
                st.session_state.openai_key = openai_key
                st.session_state.service_key = service_encoding_key
                st.success("API Key 입력 완료!")
    

    os.environ["OPENAI_API_KEY"] = st.session_state.openai_key

    st.sidebar.markdown("---")

    with st.sidebar:
        st.header("분석 옵션 선택")
        region_codes = load_region_codes()
        region = st.sidebar.selectbox("지역 선택", sorted(region_codes.keys()))
        rent_type = st.sidebar.radio("거래 유형", ["전세", "월세"])
        months_to_load = st.sidebar.slider("조회 개월 수", 3, 12, 6)

    st.write(f"선택 지역 : {region}")
    st.write(f"선택 유형 : {rent_type}")
    st.write(f"선택 개월 수 : {months_to_load}")

    if st.button("데이터 불러오기 및 분석"):
        with st.spinner("데이터 수집 중..."):
            lawd_cd = region_codes[region][:5]
            months = get_recent_months(months_to_load)
            df_all = asyncio.run(fetch_all_data(service_encoding_key, lawd_cd, months))

            if df_all.empty:
                st.warning("해당 기간 및 지역에 대한 데이터가 없습니다.")
                return

            df_filtered = process_data(df_all, rent_type)

            if not df_filtered.empty:
                st.subheader(f"📈 {region} {rent_type} 평균 보증금, 월세 추이")
                st.pyplot(plot_trend(df_filtered))
                st.dataframe(df_filtered.head(20))

                try:
                    st.session_state.chatbot = make_langgraph_chatbot(df_filtered)
                    st.success("챗봇 준비 완료! 질문해보세요.")
                    st.session_state.messages = []
                except Exception as e:
                    st.error(f"챗봇 생성 오류: {e}")
            else:
                st.warning("데이터가 없습니다.")

    st.markdown("---")
    st.subheader("🦜 데이터 기반 질문하기")

    if "chatbot" not in st.session_state:
        st.info("먼저 데이터를 불러와주세요.")
        return

    for msg in st.session_state.get("messages", []):
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    user_input = st.chat_input("예: 가장 비싼 거래는?")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("답변 생성 중..."):
            try:
                result = st.session_state.chatbot.invoke({"input": user_input})
                reply = result["output"]
                st.chat_message("assistant").markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"질문 오류: {e}")

if __name__ == "__main__":
    main()
