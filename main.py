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

@app.get("/privacy")
async def privacy_policy():
    # Method 1: Serve a static file (e.g., HTML or TXT)
    #return await quart.send_from_directory(directory='path_to_directory', filename='privacy_policy.html')

    # Method 2: Return a string
    privacy_text = """
    Privacy Policy for the LöplabbetGPT API calls

Last Updated: 2023-11-18

1. Introduction

Welcome to the LöplabbetGPT API. We are committed to respecting your privacy and protecting your personal information. This Privacy Policy explains the types of information we collect through our API and how we use this information.

2. Information We Collect

The LöplabbetGPT API is designed to collect only the information that is necessary to provide our service. The only data we collect is:

Input Data: This includes the query argument that you provide when you make an API call. We use this data solely for the purpose of processing your request and providing you with the relevant response.
3. Use of Collected Information

The information we collect is used in the following ways:

To process and respond to the queries received via the API.
To improve the functionality and performance of our API.
We do not use the information collected for any other purpose, nor do we share it with any third parties.

4. Data Retention

We retain the collected data only as long as necessary to provide the requested service. Once the purpose of data collection is fulfilled, we ensure the data is securely deleted from our systems.

5. Security

We take the security of your data seriously and implement appropriate technical and organizational measures to protect it against unauthorized or unlawful access, alteration, disclosure, or destruction.

6. Changes to This Privacy Policy

We may update this Privacy Policy from time to time. We encourage you to review this policy periodically to stay informed about how we are protecting the information we collect.

7. Contact Us

If you have any questions or concerns about this Privacy Policy or our data practices, please contact us at [Your Contact Information].
"""
    return quart.Response(response=privacy_text, status=200, mimetype='text/plain')


def main():
    app.run(debug=True, host="0.0.0.0", port=5003)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    app.run(debug=True, host="0.0.0.0", port=port)
