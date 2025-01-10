from langchain_community.llms import Ollama

q = input("What is your question?\n> ")
llm = Ollama(model="llama3.2")
res = llm.invoke(q)
print(res)
