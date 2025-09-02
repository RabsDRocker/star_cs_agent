# Filename: reasoning_api_server.py
import os
import json
import numpy as np
from typing import Dict, List, Any
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity

# Correct, compatible LangChain imports for Python 3.8
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser # This was missing before

# --- 1. SETUP AND CONFIGURATION ---
# Define the absolute base path to your package's data directory
DATA_DIR = os.path.expanduser('~/colcon_ws/src/star_cs_agent/data')

# Load the API key from the .env file in the data directory
env_path = os.path.join(DATA_DIR, '.env')
load_dotenv(dotenv_path=env_path)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(f"FATAL ERROR: OPENAI_API_KEY not found in .env file. Checked path: {env_path}")

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# --- (Load Digital Twin Data using the correct base path) ---
ALPHA = 0.7
KNOWLEDGE_BASE_PATH = os.path.join(DATA_DIR, 'knowledge_base.json')
SITE_PLAN_PATH = os.path.join(DATA_DIR, 'site_plan.json')
REPORT_FILE_PATH = os.path.join(DATA_DIR, 'safety_report.json')

def load_knowledge_base():
    with open(KNOWLEDGE_BASE_PATH, 'r') as f:
        db = json.load(f)
    embeddings = embedding_model.embed_documents([doc['text'] for doc in db])
    return db, embeddings

KNOWLEDGE_DB, KB_EMBEDDINGS = load_knowledge_base()

with open(SITE_PLAN_PATH, 'r') as f:
    SITE_PLAN = json.load(f)

# --- 2. AGENT LOGIC (WITHOUT LANGGRAPH) ---
def synthesize_query(event: Dict[str, Any]) -> str:
    print("--- 🧠 Agent Step 1: Synthesizing Query ---")
    zone_id = event.get('zone_id', 'Unknown')
    query = f"Analyze the following event: A '{event.get('event_type')}' occurred for agent '{event.get('agent_id')}' in zone '{zone_id}'. Details: {event.get('details')}"
    return query

def retrieve_context_with_sarag(event: Dict[str, Any], query: str) -> str:
    print("--- 🧠 Agent Step 2: Spatially-Aware Retrieval (SA-RAG) ---")
    zone_id = event.get('zone_id', 'Unknown')
    relevant_tags = SITE_PLAN.get(zone_id, {}).get("context_tags", [])
    filtered_indices = [i for i, doc in enumerate(KNOWLEDGE_DB) if any(tag in doc['tags'] for tag in relevant_tags)]
    
    if not filtered_indices: return "No spatially relevant regulations found."

    query_embedding = embedding_model.embed_query(query)
    filtered_embeddings = [KB_EMBEDDINGS[i] for i in filtered_indices]
    similarities = cosine_similarity([query_embedding], filtered_embeddings)[0]
    top_indices = [filtered_indices[i] for i in np.argsort(similarities)[-3:][::-1]]
    
    context = "Top Retrieved Regulations:\n"
    for idx in top_indices: context += f"- Rule: {KNOWLEDGE_DB[idx]['id']}\n  Text: {KNOWLEDGE_DB[idx]['text']}\n"
    return context

def generate_report(query: str, context: str) -> str:
    print("--- 🧠 Agent Step 3: Analysis & Recommendation ---")
    prompt = PromptTemplate.from_template(
        "You are an expert AI safety agent. Analyze a safety event and provide a report.\n"
        "**Event Data:**\n{query}\n\n"
        "**Relevant OSHA Regulations:**\n{context}\n\n"
        "**Your Task:**\n1. List potential OSHA violations.\n2. Provide a prioritized, actionable checklist for the site manager.\n"
        "Format your response with '### Violations' and '### Recommendations' sections."
    )
    chain = prompt | llm | StrOutputParser()
    report_content = chain.invoke({"query": query, "context": context})
    return report_content

# --- 3. FLASK API SERVER ---
app_flask = Flask(__name__)

@app_flask.route("/analyze", methods=['POST'])
def analyze_event():
    print("\n--- Received API request ---")
    event_data = request.json
    if not event_data: return jsonify({"error": "Invalid input"}), 400

    synthesized_query = synthesize_query(event_data)
    retrieved_context = retrieve_context_with_sarag(event_data, synthesized_query)
    final_report_content = generate_report(synthesized_query, retrieved_context)
    
    parts = final_report_content.split("###")
    violations = parts[1].replace("Violations", "").strip() if len(parts) > 1 else "N/A"
    recommendations = parts[2].replace("Recommendations", "").strip() if len(parts) > 2 else "N/A"
    report = {
        "timestamp": event_data.get('timestamp'), "event_type": event_data.get('event_type'),
        "details": event_data.get('details'), "identified_violations": violations,
        "actionable_recommendations": recommendations
    }
    with open(REPORT_FILE_PATH, "w") as f: json.dump(report, f, indent=2)
    print("--- Workflow complete. Report saved. ---")
    return jsonify({"status": "success", "report_timestamp": report['timestamp']})

# --- 4. RUN THE SERVER ---
if __name__ == "__main__":
    print("🚀 Starting STAR-CS Reasoning API Server (Final, Compatible Version)...")
    app_flask.run(host='0.0.0.0', port=5001)