from sentence_transformers import SentenceTransformer, util
import torch
import json
import sys

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
        "My name is and I am your assistant",
        "Merhaba, size nasıl yardımcı olabilirim",
        "Müşteri hizmetlerine hoş geldiniz",
        "Bu görüşme kalite kontrol amacıyla kaydedilmektedir",
        "Bankasından arıyorum",
        "Sana nasıl yardımcı olabilirim",
        "Size nasıl yardımcı olabilirim",
        "Bizi aradığınız için teşekkür ederiz",
        "Bilgilerinizi güncelleyeceğim",
        "Sisteme yansıdıktan sonra",
        "Kimlik kartınızın son kullanma tarihi",
    ]
    
    customer_anchors = [
        "I'm calling about a problem",
        "I need help with my account",
        "My order hasn't arrived yet",
        "I would like to complain about",
        "Bir sorunum var",
        "Yardım istiyorum",
        "Siparişim gelmedi",
        "Şikayette bulunmak istiyorum",
        "Bankayı aradım",
        "Şube değiştirmek istiyorum",
        "Sinirlendim",
        "En yakın şube nerede",
        "Taşındım",
        "Yeni adresim",
        "Teşekkür ederim",
        "Ne yapmam gerekiyor",
    ]

    # 3. Encode the anchors into vectors
    agent_vecs = model.encode(agent_anchors, convert_to_tensor=True)
    customer_vecs = model.encode(customer_anchors, convert_to_tensor=True)

    # 4. Grab the first significant phrase from each speaker
    # (We usually check the first 2-3 turns to be safe)
    results = {}
    
    # Initialize speakers dynamically
    for entry in transcript_data[:6]:
        speaker = entry['speaker']
        if speaker not in results:
            results[speaker] = 0.0

    for entry in transcript_data[:6]:
        speaker = entry['speaker']
        text = entry['text']
        
        if not text or len(text.strip()) < 5:
            continue
            
        text_vec = model.encode(text, convert_to_tensor=True)

        # Compare to anchors
        agent_score = torch.max(util.cos_sim(text_vec, agent_vecs)).item()
        customer_score = torch.max(util.cos_sim(text_vec, customer_vecs)).item()

        # Add to the speaker's total "Agent-ness"
        results[speaker] += (agent_score - customer_score)

    # 5. Final Verdict
    if not results:
        return {"agent": "unknown", "customer": "unknown"}
    
    agent = max(results, key=results.get)
    speakers = list(results.keys())
    customer = speakers[1] if len(speakers) > 1 and agent == speakers[0] else speakers[0] if len(speakers) > 1 else "unknown"

    return {"agent": agent, "customer": customer}


if __name__ == "__main__":
    # Eğer komut satırından JSON dosya yolu verilmişse oku
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
        with open(json_file_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        result = identify_speaker_roles(transcript_data)
        print(json.dumps(result, ensure_ascii=False))
    else:
        # Örnek kullanım
        sample_transcript = [
            {"speaker": "Speaker_0", "text": "Hello, thank you for calling Acme Corp, my name is Alex."},
            {"speaker": "Speaker_1", "text": "Hi Alex, I'm calling because my internet is down."}
        ]
        print(identify_speaker_roles(sample_transcript))