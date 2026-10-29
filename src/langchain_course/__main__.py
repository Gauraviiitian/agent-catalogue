import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from src.langchain_course.react_agent.react_agent import call_react_agent

load_dotenv()


def get_prompt_template() -> tuple[PromptTemplate, str]:
    info = """Krishna (Sanskrit: कृष्ण, IAST: Kṛṣṇa Sanskrit: [kr̩ʂɳɐ] ⓘ) also known as Govinda, Madhava, Gopala, and other names and titles is a major deity in Hinduism. He is worshipped as the eighth avatar of Vishnu and as the Supreme God in his own right. He is widely revered for his divine qualities of love, compassion, protection, and tenderness.[11] Krishna's birthday is celebrated every year by Hindus on Krishna Janmashtami, according to the lunisolar Hindu calendar, which falls in late August or early September of the Gregorian calendar.
        Krishna is often depicted in dark skin in accordance with meaning of his name, "black" or "dark". The anecdotes and narratives of Krishna's life are known as Krishna Lila. He is a central figure in the Mahabharata, the Bhagavata Purana, the Brahma Vaivarta Purana, and the Bhagavad Gita, and is mentioned in many Hindu philosophical, theological, and mythological texts.[12] These texts describes him as a divine statesman, teacher, savior of dharma, charioteer, and sometimes a cowherd. They portray him in various perspectives: as a god-child, a prankster, a model lover, a divine hero, and the universal supreme being.[13] His iconography reflects these legends and shows him in different stages of his life, such as an infant eating butter, a young boy playing a flute, a handsome youth with Radha or surrounded by female devotees, or a friendly charioteer giving counsel to Arjuna.[14]
        Krishna was born in Mathura to Devaki and Vasudeva, but was raised by Nanda and Yashoda in Gokul to escape his maternal uncle, the tyrant king Kamsa. His life, as given in texts, incorporates highly symbolic miraculous elements: he enchanted the gopis with his divine flute playing, performed the celestial Rasa Lila, and lifted Govardhan Hill to protect his community from devastating rains. He later defeated Kamsa and restored order in Mathura. He went on to establish the city of Dvaraka and played a central role in the Kurukshetra War, serving as Arjuna’s charioteer and delivering the philosophical teachings of the Bhagavad Gita.
        The name and synonyms of Krishna have been traced to 1st millennium BCE literature and cults.[15] In some sub-traditions within Hinduism, such as Krishnaism, Krishna is worshipped as the Supreme God and Svayam Bhagavan (God Himself) as he was master in all 64 forms of Art and 14 Vidya.[16] These sub-traditions arose in the context of the medieval-era Bhakti movement.[17][18] Krishna-related literature has inspired numerous performance arts such as Bharatanatyam, Kathakali, Kuchipudi, Odissi, and Manipuri dance.[19][20] He is a pan-India god, but is particularly revered in some locations, such as Vrindavan in Uttar Pradesh, Dwarka and Junagadh in Gujarat, the Jagannatha aspect in Odisha, Mayapur in West Bengal,[17][21] and other places. He is also worshipped in different regional forms across India.[note 3]
        Since the 1960s, the worship of Krishna has also spread to the Western world, largely due to the work of the Bhaktivedanta Swami and his organisation, ISKCON."""

    summary_template = """
    Summarize the following information about Krishna in 3 sentences: {info}. Also define him in 1 sentence.
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["info"],
        template=summary_template,
    )

    return summary_prompt_template, info


def call_groq() -> None:
    prompt, info = get_prompt_template()

    groq_key = os.environ.get("GROQ_API_KEY")
    llm = ChatGroq(
        model="qwen/qwen3.8-27b",
        api_key=groq_key,
        temperature=0.5,
    )

    chain = prompt | llm
    summary = chain.invoke({"info": info})
    print(summary.content)


def call_ollama() -> None:
    prompt, info = get_prompt_template()
    llm = ChatOllama(model="gemma3:270m", temperature=0.5)
    chain = prompt | llm
    summary = chain.invoke({"info": info})
    print(summary.content)


def main() -> None:
    # call_groq()
    # call_ollama()
    call_react_agent()


if __name__ == "__main__":
    main()
