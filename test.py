import openai
import streamlit as st
from tenacity import retry, wait_random_exponential
import asyncio
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from langchain.chains.llm import LLMChain
from pydantic import BaseModel
import json
from qdrant_client.models import ScoredPoint
from qdrant_client import AsyncQdrantClient
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import AsyncQdrantClient
from qdrant_client import models
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = "gpt-4o-mini-2024-07-18"
collection_name = "mecfs-rag"
qdrant_client_url=st.secrets["QDRANT_CLIENT_URL"]
qdrant_client_api_key=st.secrets["QDRANT_CLIENT_API"]
qdrant_client = AsyncQdrantClient(
    url=qdrant_client_url, 
    api_key=qdrant_client_api_key,
    prefer_grpc=True,
)
embeddings = OpenAIEmbeddings()


class LineListOutputParser(BaseOutputParser[list[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> list[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines

class HypAnswerResponse(BaseModel):
    question: str
    answer: str


class Retriever:
    async def search(self, query, top_k=10):
        query_embedding = await asyncio.to_thread(embeddings.embed_query, query)
        
        result = await qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=2.0,
                )
            ),
            limit=top_k,
            timeout=1000,
        )
        return result
    

class RagBot:
    def __init__(self, retriever, llm, model: str):
        self._retriever = retriever
        self._client = openai.Client()
        self._model = model
        self._llm = llm

    @retry(wait=wait_random_exponential(min=1, max=120))
    def completion_with_backoff(self, **kwargs):
        print("Attempting API call...")
        return self._client.chat.completions.create(**kwargs)    

    def retrieve_docs(self, question):
        return self._retriever.search(question)

    def invoke_llm(self, question, docs):
        formatted_docs = "\n\n".join([f"Document {i+1}:\n{doc.payload['text']}" for i, doc in enumerate(docs)])

        response = self.completion_with_backoff(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI research assistant with expertise in ME/CFS. "
                    "Use the following docs to produce a concise answer based on existing research on the ME/CFS field to the user question.\n\n"
                    "If you don't know the answer, say that you couldn't find information about that topic in the documents."
                    f"## Docs\n\n{formatted_docs}",
                },
                {"role": "user", "content": question},
            ],
        )

        # Evaluators will expect "answer" and "contexts"
        return {
            "answer": response.choices[0].message.content,
            "contexts": [str(doc) for doc in docs],
        }
    

    def generate_multiple_queries(
        self, question: str, include_orig_question: bool
    ) -> list[str]:
        """Generate queries based upon user input.

        Args:
            question: user query

        Returns:
            List of LLM generated queries that are similar to the user input
        """
        output_parser = LineListOutputParser()

        prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is 
            to generate 3 different versions of the given user 
            question to retrieve relevant documents from a vector  database. 
            By generating multiple perspectives on the user question, 
            your goal is to help the user overcome some of the limitations 
            of distance-based similarity search. Provide these alternative 
            questions separated by newlines. Original question: {question}""",
        )

        llm_chain = prompt | self._llm | output_parser

        response = llm_chain.invoke(
            {"question": question}
        )
        if isinstance(llm_chain, LLMChain):
            lines = response["text"]
        else:
            lines = response

        if include_orig_question:
            lines = [question] + lines    
        logger.info(f"Generated queries: {lines}")
        
        return lines

    async def generate_hypothetical_answers(self, queries: list[str]) -> list[str]:
        hyp_answers: list[str] = []
        client = AsyncOpenAI()

        async def process_query(query):
            completion = await client.beta.chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant answering questions about ME/CFS. Please answer the user questions by repeating the question and giving a precise answer"},
                    {"role": "user", "content": query}
                ],
                response_format=HypAnswerResponse,
            )
            response = completion.choices[0].message.content
            parsed_response = json.loads(response)
            return parsed_response['answer']

        tasks = [process_query(query) for query in queries]
        hyp_answers = await asyncio.gather(*tasks)

        return hyp_answers
  

    async def perform_search(self, hyp_answers: list[str]) -> list[ScoredPoint]:
        async def retrieve_docs(answer):
            return await self._retriever.search(answer)
        
        tasks = [retrieve_docs(answer) for answer in hyp_answers]
        search_results = await asyncio.gather(*tasks)

        docs = [doc for result in search_results for doc in result]

        return docs 
 

    async def get_answer(self, question: str):

        #Create multiple queries
        queries: list[str] = self.generate_multiple_queries(question=question, include_orig_question=False)
        #print(queries)

        #Create hypothetical answers for each query
        hyp_answers: list[str] = await self.generate_hypothetical_answers(queries)
        print(hyp_answers)

        #For each hyp. answer, perform search -> list of ScoredPoints, with id, score, payload={"text": "...", "metadata": {"doi": "...", "pubmed_id": "...", "keywords": "y,x,s", "authors": "name1, name2", "pub_date": "2023"}}
        docs: list[ScoredPoint] = await self.perform_search(hyp_answers)
        
        #rank the ScoredPoints by the score that is already stored in the Documents
        sorted_docs = sorted(docs, key=lambda x: x.score, reverse=True)
        seen_dois = set()

        # Liste für die gefilterten Dokumente
        unique_docs = []

        # Iteriere über die sortierten Dokumente und füge nur Dokumente mit einzigartigem DOI hinzu
        for doc in sorted_docs:
            doi = doc.payload['metadata']['doi']
            if doi and doi not in seen_dois:
                unique_docs.append(doc)
                seen_dois.add(doi)

        # Nimm nur die Top 5 einzigartigen Dokumente
        top_5_docs = unique_docs[:5]
        
        # print("Top 5 documents:")
        # for i, doc in enumerate(top_5_docs, 1):
        #     print(doc)
        #     print(f"{i}. Score: {doc.score:.4f}, DOI: {doc.payload['metadata']['doi']}")

        #pass top 5 docs to generator method
        final_answer = self.invoke_llm(question, top_5_docs)
        return {
            "answer": final_answer["answer"],
            "contexts": [doc.payload for doc in top_5_docs]
        }

    
async def rag():
    retriever = Retriever()
    llm = ChatOpenAI(model=model, temperature=0)
    rag_bot = RagBot(retriever, llm, model)
    

    answer_info = await rag_bot.get_answer(prompt)

    answer_text = answer_info["answer"]


   # Sammlung der Quellen als Text
    sources_text = ""
    
    for doc in answer_info.get("contexts", []):
        metadata = doc.get("metadata", {})
        title = metadata.get("title", "No title available")
        doi = metadata.get("doi")
        
        # Autoren als String holen und in eine Liste umwandeln
        authors = metadata.get("authors", "")
        authors_list = [author.strip() for author in authors.split(",")] if authors else []
        
        # Wenn mehr als 5 Autoren, kürzen und "..." hinzufügen
        if len(authors_list) > 5:
            authors_list = authors_list[:5] + ["..."]
        
        # Die Autoren als String zusammenfügen
        authors_text = ", ".join(authors_list)
        
        # Wenn ein DOI vorhanden ist, als Link einfügen
        if doi:
            # DOI-Link korrekt formatieren
            doi_link = f"https://doi.org/{doi}"
            sources_text += f"📚 [{title}]({doi_link}) - {authors_text} ({metadata.get('pub_date', 'Unknown date')})\n"
        else:
            sources_text += f"📚 {title} - {authors_text} ({metadata.get('pub_date', 'Unknown date')})\n"
    
    # Die Antwort und Quellen in einer einzigen Nachricht kombinieren
    full_response = answer_text
    if sources_text:
        full_response += f"\n\n**Quellen:**\n{sources_text}"


    # Die gesamte Antwort (inklusive Quellen) in einer Nachricht anzeigen
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.chat_message("assistant").write(full_response)
# Setup für Streamlit
openai_api_key = st.secrets["OPENAI_API_KEY"]
st.title("💬 Chatbot")
st.caption("Find answers about ME/CFS in published studies.")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Was möchtest du über ME/CFS wissen?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Wenn eine Eingabe kommt, verwende RagBot, um die Antwort zu erhalten
if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.spinner("Searching and generating answer..."):
        asyncio.run(rag())
