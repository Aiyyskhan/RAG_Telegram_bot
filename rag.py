import os
import pandas as pd
import numpy as np
import openai
from scipy import spatial
from collections import deque
from dotenv import load_dotenv

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_KEY")

# емкость памяти об истории чата
chat_history_maxlen = 10

# системный промпт
SYS_PROMPT = (
    "Ты бот-помощник в задачах, связанных с ответами на вопросы. "
    "При первом знакомстве представляйся и кратко расскажи кто ты такой. "
    "Ничего не выдумывай. "
    "Если не знаешь ответа, скажи, что не знаешь. "
    # "Используй максимум четыре предложения и будь краток. "
    "Пиши вежливо, красиво, делай абзацы с отступами сверху и снизу. "
    "При приветствии учитывай временную зону, которая +05:00. "
    "\n\n"
    "Ссылка на источник статьи: https://openai.com/index/evolution-strategies/ "
    "\n\n"
    "Документ: {context} "
    "\n\n"
    "Твоё имя: {bot_name} "
    "\n\n"
)

client = openai.OpenAI(api_key=OPENAI_KEY)

# загрузка документа для базы знания
df_3 = pd.read_csv("Evolution_Strategies_as_a_scalable_alternative_to_Reinforcement_embed.csv", index_col=0)
df_3["embeddings"] = df_3["embeddings"].apply(eval).apply(np.array)

# text_series = df_3.iloc[:, 1].astype("str")
# text_string = " ".join(text_series)

def distances_from_embeddings(query_embedding, embeddings, dist_metric = "cosine"):
    """
    Функция вычисления расстояния эмбеддингов
    """
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev
    }
    dist = [distance_metrics[dist_metric](query_embedding, embedding) for embedding in embeddings]
    return dist

def create_context(question, df, max_len = 1000):
    """
    Функция извлечения контекста из базы знания в зависимости от запроса
    """
    q_embeddings = client.embeddings.create(input=question, model="text-embedding-3-large").data[0].embedding

    df["distances"] = distances_from_embeddings(q_embeddings, df["embeddings"].values)

    returns = []
    cur_len = 0

    for i, row in df.sort_values("distances", ascending=True).iterrows():
        cur_len += row["n_tokens"] + 4
        if cur_len > max_len:
            break
        returns.append(row["text"])

    return "\n\n###\n\n".join(returns)

# chat_history = {}
def answer_question(
        question,
        system_prompt,
        df,
        session_id = 0,
        bot_name = "Bot",
        model = "gpt-3.5-turbo", #"gpt-4-turbo",
        max_len = 3500,
        max_tokens = 4000,
        debug = False,
        stop_sequence = None,
        chat_history = None
    ):
    """
    Функция ввода-вывода по API LLM
    """

    # if session_id not in chat_history:
    #     chat_history[session_id] = ""

    try:
        # context_q = client.chat.completions.create(
        #     model=model,
        #     messages=[
        #         {"role": "system", "content": "ты - серьезный робот-архивариус, генерирующий информацию. Не отвечай, просто генерируй cухую фактическую информацию."},
        #         # {"role": "user", "content": f"History: {chat_history[session_id]}\nQuestion: {question}\n"},
        #         *([] if chat_history is None else list(chat_history)),
        #         # {"role": "user", 
        #         #     "content": "учитывая приведенный выше разговор и историю разговора, сгенерируйте поисковый запрос, чтобы найти информацию, относящуюся к разговору"
        #         # }
        #         {"role": "user", 
        #             "content": "сгенерируй короткую выдержку, выделив самые важные, ключевые моменты беседы."
        #         }
        #     ],
        #     temperature=0.0,
        #     max_tokens=max_tokens,
        #     top_p=1,
        #     frequency_penalty=0,
        #     presence_penalty=0,
        #     stop=stop_sequence,
        #     # stream=True
        # )

        # search_q = context_q.choices[0].message.content
        search_q = question

        # print(f"Search q:\n{search_q}")
        # print("\n\n")

        context = create_context(
            search_q,
            df,
            max_len=max_len
        )

        if debug:
            print(f"Search q:\n{search_q}")
            print("\n\n")
            print(f"Context:\n{context}")
            print("\n\n")
            print(f"History:\n{chat_history}")
            print("\n\n")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt.format(context=context, bot_name=bot_name)},
                *([] if chat_history is None else list(chat_history)),
                {"role": "user", "content": f"Вопрос: {question}\nОтвет:"}
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
        )

        answer = response.choices[0].message.content

        # hist_resp = client.chat.completions.create(
        #     model=model,
        #     messages=[
        #         {"role": "system", "content": "ты - архивариус, генерирующий информацию для истории чата. Не отвечай, просто генерируй сжатую информацию."},
        #         # *([] if chat_history is None else list(chat_history)),
        #         {"role": "user", "content": f"История беседы: {chat_history[session_id]}\nВопрос: {question}\nОтвет:"},
        #         {"role": "assistant", "content": answer},
        #         {"role": "user", 
        #             "content": "сгенерируй короткую выдержку, выделив самые важные, ключевые моменты беседы."
        #         }
        #     ],
        #     temperature=0.0,
        #     max_tokens=max_tokens,
        #     top_p=1,
        #     frequency_penalty=0,
        #     presence_penalty=0,
        #     stop=stop_sequence,
        #     # stream=True
        # )

        # chat_history[session_id] = hist_resp.choices[0].message.content
        
        if chat_history is not None:
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": answer})

        return answer
    except Exception as e:
        print(e)
        return ""

chat_history = {}
def get_session_history(session_id):
    """
    Функция памяти об истории чата
    """
    if session_id not in chat_history:
        chat_history[session_id] = deque(maxlen=chat_history_maxlen)
    return chat_history[session_id]

def main(query, session_id, bot_name):
    # return answer_question(query, SYS_PROMPT, df_3, session_id, bot_name, user_from)
    return answer_question(
        question=query, 
        system_prompt=SYS_PROMPT, 
        df=df_3,
        session_id=session_id,
        bot_name=bot_name,
        debug=True,
        chat_history=get_session_history(session_id)
    )
