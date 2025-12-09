# STAR-CS Reasoning API Server with Spatially-Aware RAG and Rich Risk-Aware Reasoning

import os
import json
from typing import Dict, List, Any

import numpy as np
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -------- Paths and configuration --------

DATA_DIR = os.path.expanduser('~/colcon_ws/src/star_cs_agent/data')
env_path = os.path.join(DATA_DIR, '.env')
load_dotenv(dotenv_path=env_path)

if not os.getenv('OPENAI_API_KEY'):
    raise ValueError(f"FATAL ERROR: OPENAI_API_KEY not found in .env file. Checked path: {env_path}")

llm = ChatOpenAI(model='gpt-4o', temperature=0.1)
embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')

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

# -------- Spatially-Aware RAG helpers --------


def synthesize_query(event: Dict[str, Any]) -> str:
    """
    Turn raw event JSON into a rich natural-language description for the LLM,
    including:
    - Zone entry context (hazards + agents)
    - High-risk zones (from risk field)
    - Predicted interactions with cell IDs and positions.
    """
    etype = event.get('event_type', 'Unknown Event')
    lines = [f"Event type: {etype}.", f"Timestamp: {event.get('timestamp')}.\n"]

    # 1) Zone entry narrative
    agent_id = event.get('agent_id')
    zone_id = event.get('zone_id')
    if agent_id:
        lines.append(f"Primary agent: {agent_id}.")
    if zone_id and zone_id in SITE_PLAN:
        desc = SITE_PLAN[zone_id].get('description', '')
        lines.append(f"Zone: {zone_id} — {desc}.")

    details = event.get('details')
    if details:
        lines.append(f"Details: {details}")

    # Zone-level hazards and agents for Zone Entry events
    zone_hazards = event.get('zone_hazards') or []
    zone_agents = event.get('zone_agents') or []
    if zone_hazards or zone_agents:
        lines.append("\nLocal zone composition at the time of entry:")
        if zone_hazards:
            hz_names = [f"{hz['class']} (id={hz['hazard_id']})" for hz in zone_hazards]
            lines.append(f"- Hazards present in this zone: {', '.join(hz_names)}.")
        if zone_agents:
            descs = []
            for a in zone_agents:
                descs.append(
                    f"{a['agent_id']} ({a['agent_type']}, speed≈{a['speed_m_s']:.2f} m/s)"
                )
            lines.append(f"- Agents currently in this zone: {', '.join(descs)}.")

    # 2) High-risk zones (from risk field snapshots)
    risk_zones = event.get('risk_zones') or []
    if risk_zones:
        lines.append("\nHigh-risk zones based on current risk topography:")
        for idx, rz in enumerate(risk_zones[:5], start=1):
            zid = rz.get('zone_id')
            score = rz.get('total_risk', 0.0)
            maxv = rz.get('max_risk', 0.0)
            num_cells = rz.get('num_cells', 0)
            agents = rz.get('agents', [])
            hz_details = rz.get('hazard_details', [])
            hz_names = [h.get('class', 'unknown') for h in hz_details]
            lines.append(
                f"  {idx}) Zone {zid}: total risk {score:.2f}, max cell risk {maxv:.2f}, "
                f"{num_cells} hot cells, agents={agents}, hazards={hz_names}."
            )

    # 3) Predicted interactions (agent-agent and agent-hazard) with cell IDs
    inters = event.get('predictive_interactions') or []
    if inters:
        lines.append("\nPredicted risky interactions (based on velocity, pose, and TTC):")
        for inter in inters[:5]:
            itype = inter.get('type')
            participants = inter.get('agents', [])
            ttc = inter.get('ttc')
            cell_id = inter.get('cell_id')
            cell_x = inter.get('cell_x')
            cell_y = inter.get('cell_y')
            hz_class = inter.get('hazard_class')
            if itype == "agent-hazard" and hz_class:
                lines.append(
                    f"- {itype} between {participants[0]} and hazard '{hz_class}' "
                    f"with estimated time-to-collision ~{ttc:.1f}s near cell {cell_id} "
                    f"at approx. ({cell_x:.2f}, {cell_y:.2f})."
                )
            else:
                lines.append(
                    f"- {itype} between {participants} with estimated time-to-collision ~{ttc:.1f}s "
                    f"near cell {cell_id} at approx. ({cell_x:.2f}, {cell_y:.2f})."
                )

    # 4) Agent locations (global)
    all_agents = event.get('all_agents_locations') or {}
    if all_agents:
        lines.append("\nCurrent agent locations (global coordinates):")
        for aid, loc in all_agents.items():
            lines.append(f"- {aid}: ({loc['x']:.2f}, {loc['y']:.2f}, {loc['z']:.2f})")

    # NOTE: Even though we don't persist long-term memory here,
    # we explicitly *tell* the LLM to behave as if it has long-term
    # spatial-temporal memory of similar events and risk patterns.
    lines.append(
        "\nAssume you are a maturing safety agent with long-term spatial and temporal memory "
        "of this site (zones, cells, hazards, occupancy, and past events), similar to an "
        "experienced safety manager who has been watching this site for a long time."
    )

    return "\n".join(lines)


def retrieve_context_with_sarag(event: Dict[str, Any], query: str) -> str:
    """
    Spatially-Aware RAG:
    - Identify relevant zones (zone_id + risk_zones)
    - Collect their context_tags from site_plan
    - Filter OSHA/NIOSH docs by those tags
    - Run embedding similarity within that subset
    """
    # collect all zone ids referenced in the event
    zone_ids: List[str] = []
    if 'zone_id' in event and event['zone_id']:
        zone_ids.append(event['zone_id'])
    for rz in event.get('risk_zones', []):
        zid = rz.get('zone_id')
        if zid:
            zone_ids.append(zid)
    zone_ids = list({z for z in zone_ids if z in SITE_PLAN})  # unique & valid

    context_tags = set()
    for zid in zone_ids:
        tags = SITE_PLAN[zid].get('context_tags', [])
        context_tags.update(tags)

    # fall back to whole DB if we have no tags
    if context_tags:
        candidate_indices = [
            i for i, doc in enumerate(KNOWLEDGE_DB)
            if set(doc.get('tags', [])) & context_tags
        ]
    else:
        candidate_indices = list(range(len(KNOWLEDGE_DB)))

    if not candidate_indices:
        return "No spatially relevant OSHA/NIOSH regulations found for the referenced zones."

    query_emb = embedding_model.embed_query(query)
    cand_embs = np.array([KB_EMBEDDINGS[i] for i in candidate_indices])
    sims = cosine_similarity([query_emb], cand_embs)[0]

    top_k = 3
    top_local_idx = np.argsort(sims)[-top_k:][::-1]
    context_lines = ["Top Retrieved Regulations (spatially filtered OSHA/NIOSH clauses):"]
    for li in top_local_idx:
        gi = candidate_indices[int(li)]
        doc = KNOWLEDGE_DB[gi]
        context_lines.append(f"- Rule: {doc.get('id')}\n  Text: {doc.get('text')}")
    return "\n".join(context_lines)


def generate_report(query: str, context: str) -> str:
    """
    Ask the LLM to:
    - Summarize situation (zone entry or snapshot)
    - Describe risky zones & interactions
    - Provide Safety Regulatory Concerns (OSHA/NIOSH) with predictive, physics-based reasoning
    - Provide preventive measures
    Output format has four explicit sections.
    """
    prompt = PromptTemplate.from_template(
        "You are an AI safety co-pilot for construction sites. You behave like a mature, "
        "scientific-minded safety manager with a strong understanding of physics (kinematics, "
        "time-to-collision, momentum) and safety regulations (OSHA and NIOSH).\n\n"
        "You receive:\n"
        "## Event & Risk Context\n{query}\n\n"
        "## Relevant OSHA/NIOSH Regulations\n{context}\n\n"
        "## Your Tasks\n"
        "1. Provide a concise but clear summary of what is happening in space and time.\n"
        "2. Identify which zones are currently most risky, how many agents and which hazards they contain, "
        "and describe any predicted risky interactions (who is moving toward what, approximate cell IDs/locations).\n"
        "3. Under the heading **Safety Regulatory Concerns**, reason about the regulatory situation:\n"
        "   - Use OSHA and NIOSH concepts that are relevant to this construction environment.\n"
        "   - Consider current and near-future (e.g., ~30–50 seconds) risk, based on agent-agent and agent-hazard "
        "     proximity, pose/orientation, velocity, and predicted interactions.\n"
        "   - Make physics-based, factual comments: e.g., closing speeds, likely time-to-collision, crowding/occupancy "
        "     in a zone, how these could lead to violations if nothing changes.\n"
        "   - Discuss whether there is a possibility of breaching safety regulations (not only actual violations), "
        "     and explain why.\n"
        "   - Think like a safety agent that has spatio-temporal memory of this site (zones, cells, risk topography, "
        "     past events) and refines its judgment over time.\n"
        "4. Provide a prioritized, actionable checklist of preventive measures for the safety manager, aligned with "
        "   the physics of the situation (e.g., reduce speed, increase separation distance, re-route paths, isolation, "
        "   guarding, signage, lockout/tagout, housekeeping) and the regulatory concerns above.\n\n"
        "Format your response using **exactly** these four sections (and in this order):\n"
        "### Situation Overview\n"
        "### Risk Zones & Interactions\n"
        "### Safety Regulatory Concerns\n"
        "### Recommendations\n"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query, "context": context})


def parse_report_sections(report_text: str) -> Dict[str, str]:
    """
    Parse the four expected sections from the LLM output.
    Falls back to 'N/A' if a section is missing.
    """
    sections = {
        "Situation Overview": "N/A",
        "Risk Zones & Interactions": "N/A",
        "Safety Regulatory Concerns": "N/A",
        "Recommendations": "N/A",
    }

    # Split on '###' and inspect headings
    parts = report_text.split("###")
    for raw in parts:
        text = raw.strip()
        if not text:
            continue
        lower = text.lower()
        if lower.startswith("situation overview"):
            sections["Situation Overview"] = text[len("Situation Overview"):].strip()
        elif lower.startswith("risk zones & interactions"):
            sections["Risk Zones & Interactions"] = text[len("Risk Zones & Interactions"):].strip()
        elif lower.startswith("risk zones and interactions"):
            sections["Risk Zones & Interactions"] = text[len("Risk Zones and Interactions"):].strip()
        elif lower.startswith("safety regulatory concerns"):
            sections["Safety Regulatory Concerns"] = text[len("Safety Regulatory Concerns"):].strip()
        elif lower.startswith("recommendations"):
            sections["Recommendations"] = text[len("Recommendations"):].strip()

    return sections


# -------- Flask app --------

app_flask = Flask(__name__)


@app_flask.route('/analyze', methods=['POST'])
def analyze_event():
    """Main endpoint called by the ROS2 command center."""
    try:
        event_data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Invalid JSON: {e}"}), 400

    print("\n=== Received event for analysis ===")
    print(json.dumps(event_data, indent=2))

    query = synthesize_query(event_data)
    print("--- Step 1: Synthesized query ---\n", query)

    context = retrieve_context_with_sarag(event_data, query)
    print("--- Step 2: Retrieved context ---\n", context)

    report_content = generate_report(query, context)
    print("--- Step 3: LLM report ---\n", report_content)

    sections = parse_report_sections(report_content)

    report = {
        "timestamp": event_data.get("timestamp"),
        "event_type": event_data.get("event_type"),
        "details": event_data.get("details"),
        "situation_overview": sections["Situation Overview"],
        "risk_zones_and_interactions": sections["Risk Zones & Interactions"],
        "safety_regulatory_concerns": sections["Safety Regulatory Concerns"],
        "actionable_recommendations": sections["Recommendations"],
    }

    with open(REPORT_FILE_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    print("--- Workflow complete. Report saved. ---")

    return jsonify({"status": "success", "report_timestamp": report["timestamp"]})


if __name__ == '__main__':
    print('🚀 Starting STAR-CS Reasoning API Server (Safety Regulatory Concerns version)...')
    app_flask.run(host='0.0.0.0', port=5001)
