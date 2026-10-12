import sys
import json
import urllib.request
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from trustagent.retrieval.retrieve import retrieve
from trustagent.pipeline.run import run_pipeline

def print_header(title):
    print("\n" + "=" * 65)
    print(f" [TEST] {title}")
    print("=" * 65)

def test_qdrant_retrieval():
    print_header("TEST 1: Qdrant Vector Search Retrieval")
    query = "Fatshark Teleporter headset"
    results = retrieve(query, k=1)
    if results and len(results) > 0:
        record = results[0]["record"]
        score = results[0]["score"]
        print(f"  [SUCCESS] Retrieved product '{record.get('entity')}'")
        print(f"     Match Similarity Score: {score:.4f}")
        return True
    else:
        print("  [FAILED] No results returned from Qdrant database.")
        return False

def test_true_claim():
    print_header("TEST 2: True Claim Verification (Expected: SUPPORT)")
    claim = "FS-1051 FATSHARK TELEPORTER V3 HEADSET includes a 250mW 5.8Ghz transmitter."
    res = run_pipeline(claim, {})
    t_label = res["claims"][0]["modality_scores"]["text"]["label"]
    t_score = res["claims"][0]["modality_scores"]["text"]["score"]
    
    print(f"  Claim: '{claim}'")
    print(f"  NLI Result: {t_label} (Confidence: {t_score*100:.1f}%)")
    print(f"  Pipeline Decision: {res['decision']} (Trust Score: {res['trust_score']})")
    
    if t_label == "SUPPORT":
        print("  [SUCCESS] True claim correctly identified as SUPPORT!")
        return True
    else:
        print(f"  [FAILED] True claim got {t_label} instead of SUPPORT.")
        return False

def test_false_claim():
    print_header("TEST 3: Hallucinated / False Claim (Expected: CONTRADICT or BLOCK)")
    claim = "FS-1051 FATSHARK TELEPORTER V3 HEADSET features a 8K Quantum Dot OLED display."
    res = run_pipeline(claim, {})
    t_label = res["claims"][0]["modality_scores"]["text"]["label"]
    decision = res["decision"]
    
    print(f"  Claim: '{claim}'")
    print(f"  NLI Result: {t_label}")
    print(f"  Pipeline Decision: {decision} (Trust Score: {res['trust_score']})")
    
    if t_label == "CONTRADICT" or decision in ["BLOCK", "HUMAN_REVIEW"]:
        print(f"  [SUCCESS] False claim correctly flagged as {decision}!")
        return True
    else:
        print("  [FAILED] False claim passed without warning.")
        return False

def test_live_api():
    print_header("TEST 4: Live FastAPI Server Endpoint (http://localhost:8000/verify)")
    url = "http://localhost:8000/verify"
    payload = {
        "agent_response": "FS-1051 FATSHARK TELEPORTER V3 HEADSET has an integrated NexwaveRF receiver.",
        "context": {}
    }
    try:
        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            print(f"  API Response Status: {resp.status}")
            print(f"  Returned Decision: {data.get('decision')}")
            print(f"  Returned Trust Score: {data.get('trust_score')}")
            print("  [SUCCESS] Backend REST API is alive and functioning correctly!")
            return True
    except Exception as e:
        print(f"  [FAILED] to connect to FastAPI endpoint: {e}")
        return False

if __name__ == "__main__":
    print("\n>>> RUNNING COMPLETE TRUSTAGENT SYSTEM HEALTH AUDIT...")
    
    results = [
        ("Qdrant Vector Retrieval", test_qdrant_retrieval()),
        ("NLI True Claim Verification", test_true_claim()),
        ("NLI False Claim Detection", test_false_claim()),
        ("FastAPI Backend Server", test_live_api())
    ]
    
    print("\n" + "=" * 65)
    print(" FINAL SYSTEM HEALTH SUMMARY")
    print("=" * 65)
    passed = 0
    for name, ok in results:
        status = "PASSED" if ok else "FAILED"
        if ok: passed += 1
        print(f"  * {name:<35} : {status}")
    print(f"\nOverall Result: {passed}/{len(results)} Component Tests Passed.")
    print("=" * 65 + "\n")
