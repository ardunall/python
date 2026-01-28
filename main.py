from sentence_transformers import SentenceTransformer, util
import torch

# 1. Load a tiny, fast model (only ~80MB)
model = SentenceTransformer('all-MiniLM-L6-v2')

def identify_speaker_roles(transcript_data):
    """
    transcript_data: List of dicts [{"speaker": "Speaker_0", "text": "..."}]
    """
    # 2. Define our "Anchor" intents
    agent_anchors = [
        "Thank you for calling support",
        "How can I help you today?",
        "This call may be recorded for quality purposes",
        "My name is and I am your assistant"
    ]
    
    customer_anchors = [
        "I'm calling about a problem",
        "I need help with my account",
        "My order hasn't arrived yet",
        "I would like to complain about"
    ]

    # 3. Encode the anchors into vectors
    agent_vecs = model.encode(agent_anchors, convert_to_tensor=True)
    customer_vecs = model.encode(customer_anchors, convert_to_tensor=True)

    # 4. Grab the first significant phrase from each speaker
    # (We usually check the first 2-3 turns to be safe)
    results = {"Speaker_0": 0, "Speaker_1": 0}
    
    for entry in transcript_data[:4]:
        speaker = entry['speaker']
        text = entry['text']
        text_vec = model.encode(text, convert_to_tensor=True)

        # Compare to anchors
        agent_score = torch.max(util.cos_sim(text_vec, agent_vecs)).item()
        customer_score = torch.max(util.cos_sim(text_vec, customer_vecs)).item()

        # Add to the speaker's total "Agent-ness"
        results[speaker] += (agent_score - customer_score)

    # 5. Final Verdict
    agent = max(results, key=results.get)
    customer = "Speaker_1" if agent == "Speaker_0" else "Speaker_0"

    return {"agent": agent, "customer": customer}

# Example Usage
sample_transcript = [
    {"speaker": "Speaker_0", "text": "Hello, thank you for calling Acme Corp, my name is Alex."},
    {"speaker": "Speaker_1", "text": "Hi Alex, I'm calling because my internet is down."}
]

print(identify_speaker_roles(sample_transcript))