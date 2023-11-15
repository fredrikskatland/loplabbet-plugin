import json
import pickle
import quart
import quart_cors
from quart import request
import os

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

def convert_to_dict(result):
    page_content = [doc.page_content for doc in result]
    metadatas = [doc.metadata for doc in result]

    output = []
    for i in range(len(page_content)):
        output.append({'page_content': page_content[i], 'metadata': metadatas[i]})

    return output

app = quart_cors.cors(quart.Quart(__name__))

@app.post("/retrieve")
async def retrieve():
    request = await quart.request.get_json(force=True)
    query = request["query"]
    result = retriever.get_relevant_documents(query)
    output = convert_to_dict(result)
    # Convert Documents to 
    return quart.Response(response=json.dumps(output), status=200)

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
    port = int(os.environ.get("PORT", 5003))
    app.run(debug=True, host="0.0.0.0", port=port)
