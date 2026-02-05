import os
import sys
import time
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# ----------------- API KEY CONFIGURATION -----------------
try:
    API_KEY = os.environ["GEMINI_PAID_KEY"]
except KeyError:
    print("Error: GEMINI_PAID_KEY not set.")
    sys.exit(1)

genai.configure(api_key=API_KEY)

MODEL_ID = "gemini-3-pro-preview"

# ----------------- SYSTEM INSTRUCTION FOR TOPOLOGY -----------------
SYSTEM_INSTRUCTION_TOPOLOGY =""" 
    You are a CAD topology extraction expert.
    Analyze the PROVIDED ORIGINAL CAD IMAGE (NOT recolored).

    Output STRICT JSON inside markers.
    No explanation.

    <<<TOPOLOGY_START>>>
    {
    "views": [
        {
        "view": "front",
        "outer_closed_loops": [
            {
            "loop_id": 1,
            "region": "MATERIAL",
            "nodes": 4,
            "edges": 4,
            "edge_type": {
                "straight": 4,
                "chamfer": 0,
                "arc": 0
            }
            }
        ],
        "inner_closed_loops": [
            {
            "loop_id": 2,
            "region": "VOID",
            "nodes": 0,
            "edges": 0,
            "type": "Circle"
            },
            {
            "loop_id": 2,
            "nodes": 0,
            "edges": 0,
            "type": "Concentric Circles"
            "concentric circle details":{
                "circles": 3,
                "circle 1": {
                    "type": "MATERIAL"
                }
                "circle 2": {
                    "type": "MATERIAL"
                }
                "circle 3": {
                    "type": "VOID"
                }
            },
            {
            "loop_id": 3,
            "region": "VOID",
            "nodes": 4,
            "edges": 4,
            "edge_type": {
                "straight": 2,
                "arc": 2
            }
            "type": "Slot"
            }
        ]
        }
    ]
    }
Rules :
-Inner loops are always VOID, outer loops are always MATERIAL.
-If entity is Circle: nodes = 0, edges = 0, only mention the type.
- If concentric circles are present, mention so. The innermost circle should be void and the rest should be material.
Feature to topology mapping
-Rectangle → 4 nodes, 4 straight edges
-Fillet → rounded edge replacing sharp corner
-Chamfer → angled edge replacing corner

    <<<TOPOLOGY_END>>>
    """

# ----------------- FEATURE EXTRACTION (TOPOLOGY ONLY) -----------------
def extract_topology(image_path: str, output_topology_path: str):
    print(f"\nProcessing: {image_path}")

    # ---------- UPLOAD ORIGINAL IMAGE ----------
    original_file = genai.upload_file(
        path=image_path,
        display_name="CAD_Original_For_Topology"
    )

    while original_file.state.name == "PROCESSING":
        time.sleep(1)
        original_file = genai.get_file(original_file.name)

    # ---------- GENERATE TOPOLOGY ----------
    topo_model = genai.GenerativeModel(
        model_name=MODEL_ID,
        system_instruction=SYSTEM_INSTRUCTION_TOPOLOGY,
        generation_config={"temperature": 0}
    )

    topo_response = topo_model.generate_content(
        contents=[original_file]
    )

    # ---------- SAVE TOPOLOGY TEXT ----------
    topology_text = ""
    for part in topo_response.candidates[0].content.parts:
        if part.text:
            topology_text += part.text

    with open(output_topology_path, "w", encoding="utf-8") as f:
        f.write(topology_text)

    print(f"Topology saved: {output_topology_path}")

    # ---------- CLEANUP ----------
    genai.delete_file(original_file.name)


# ----------------- MAIN -----------------
if __name__ == "__main__":
    img_in = "/Users/naswahmanandhar/Desktop/RAG/images/27.png"
    topo_out = "/Users/naswahmanandhar/Desktop/RAG/LLM topology result/27LLM.txt"

    extract_topology(img_in, topo_out)