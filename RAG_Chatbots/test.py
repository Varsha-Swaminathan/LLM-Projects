from langserve import RemoteRunnable
remote_chain=RemoteRunnable("http://localhost:8000/chain/")
result=remote_chain.invoke({"language":"french","text":"good morning"})
print(result)