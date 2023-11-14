import json
import pickle
import quart
import quart_cors
from quart import request

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Import documents.pickle
with open("documents.pickle", "rb") as f:
    documents = pickle.load(f)

# Create a Chroma database
db = Chroma.from_documents(documents, embeddings)
#db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = db.as_retriever()


app = quart_cors.cors(quart.Quart(__name__), allow_origin="https://chat.openai.com")

@app.post("/retrieve")
async def retrieve():
    request = await quart.request.get_json(force=True)
    query = request["query"]
    result = retriever.get_relevant_documents(query)
    return quart.Response(response=json.dumps(result), status=200)

@app.get("/.well-known/ai-plugin.json")
async def plugin_manifest():
    host = request.headers['Host']
    with open("./.well-known/ai-plugin.json") as f:
        text = f.read()
        return quart.Response(text, mimetype="text/json")

@app.get("/openapi.yaml")
async def openapi_spec():
    host = request.headers['Host']
    with open("openapi.yaml") as f:
        text = f.read()
        return quart.Response(text, mimetype="text/yaml")

def main():
    app.run(debug=True, host="0.0.0.0", port=5003)

if __name__ == "__main__":
    main()
